# -*- coding: utf-8 -*-
"""
PyQt6 + OpenCV face-swap with full 468-point MediaPipe landmarks.
GPU/CUDA acceleration for triangle warping using TensorFlow (ImageProjectiveTransformV3).
Notes:
 - seamlessClone remains CPU-only (OpenCV).
 - Requires: pip install pyqt6 opencv-python mediapipe numpy tensorflow
"""
from __future__ import annotations
import sys
import os
from typing import Optional, Tuple, List

import numpy as np
import cv2

# --------- Optional MediaPipe for dense landmarks ---------
import tensorflow as tf
print("TF version:", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("MediaPipe version:", mp.__version__)
except Exception as e:
    MP_AVAILABLE = False
    print("MediaPipe import hatası:", e)

# --------- PyQt6 UI ---------
from PyQt6 import QtCore, QtGui, QtWidgets

# =============================
# Utility
# =============================
def to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

# =============================
# Landmark detection
# =============================
class LandmarkDetector:
    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

class MediaPipeFaceMeshDetector(LandmarkDetector):
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. 'pip install mediapipe'.")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # GPU desteği için (MediaPipe 0.10.x sürümlerinde farklı olabilir)
        try:
            # Yeni MediaPipe API'si (0.10.5+)
            if hasattr(mp.tasks, 'BaseOptions'):
                mp.tasks.BaseOptions.delegate = mp.tasks.BaseOptions.Delegate.GPU
                print("MediaPipeFaceMesh initialized with GPU support.")
            else:
                # Eski API için
                print("MediaPipeFaceMesh initialized (GPU API not available in this version).")
        except Exception as e:
            print(f"GPU configuration warning: {e}")
            print("MediaPipeFaceMesh initialized with CPU fallback.")

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = bgr.shape[:2]
        face = result.multi_face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
        return pts

# =============================
# Smoothing (EMA)
# =============================
class EMASmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state: Optional[np.ndarray] = None

    def reset(self):
        self.state = None

    def update(self, pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pts is None:
            return None
        if self.state is None:
            self.state = pts.copy()
        else:
            self.state = self.alpha * pts + (1.0 - self.alpha) * self.state
        return self.state

# =============================
# Geometry helpers
# =============================
def rect_from_points(points: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    return x, y, w, h

def delaunay_triangulation(rect: Tuple[int,int,int,int], points: np.ndarray) -> List[Tuple[int,int,int]]:
    x, y, w, h = rect
    subdiv = cv2.Subdiv2D((x, y, x + w, y + h))
    for p in points:
        subdiv.insert(tuple(p))
    triangle_list = subdiv.getTriangleList()

    def find_index(pt):
        d = np.sum((points - pt)**2, axis=1)
        return int(np.argmin(d))

    indices = []
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if (x <= pt1[0] <= x + w and y <= pt1[1] <= y + h and
            x <= pt2[0] <= x + w and y <= pt2[1] <= y + h and
            x <= pt3[0] <= x + w and y <= pt3[1] <= y + h):
            idx = (find_index(np.array(pt1)), find_index(np.array(pt2)), find_index(np.array(pt3)))
            indices.append(idx)
    return indices

# =============================
# GPU-accelerated triangle warp using TensorFlow
# =============================
def _compute_affine_transform_mat(src_tri: np.ndarray, dst_tri: np.ndarray) -> Optional[np.ndarray]:
    ones = np.ones((3,1), dtype=np.float32)
    mat_dst = np.hstack([dst_tri, ones])  # 3x3
    mat_src = src_tri
    try:
        inv_dst = np.linalg.inv(mat_dst)
        A = (inv_dst @ mat_src).T
        return A.astype(np.float32)
    except np.linalg.LinAlgError:
        return None

def warp_triangle_tf(src_img: np.ndarray, dst_img: np.ndarray, t_src: List[np.ndarray], t_dst: List[np.ndarray]):
    r2 = cv2.boundingRect(np.float32([t_dst]))
    x2, y2, w2, h2 = r2
    if w2 == 0 or h2 == 0:
        return

    r1 = cv2.boundingRect(np.float32([t_src]))
    x1, y1, w1, h1 = r1
    if w1 == 0 or h1 == 0:
        return

    t1_rect = np.array([[t_src[i][0] - x1, t_src[i][1] - y1] for i in range(3)], dtype=np.float32)
    t2_rect = np.array([[t_dst[i][0] - x2, t_dst[i][1] - y2] for i in range(3)], dtype=np.float32)

    src_patch = src_img[y1:y1+h1, x1:x1+w1]
    if src_patch.size == 0:
        return

    A = _compute_affine_transform_mat(t1_rect, t2_rect)
    if A is None:
        return

    transform = np.array([A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], 0.0, 0.0], dtype=np.float32)

    src_patch_rgb = cv2.cvtColor(src_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tf = tf.convert_to_tensor(src_patch_rgb, dtype=tf.float32)
    img_tf = tf.expand_dims(img_tf, axis=0)

    try:
        out_shape = tf.constant([h2, w2], dtype=tf.int32)
        transforms = tf.convert_to_tensor(transform.reshape(1,8), dtype=tf.float32)
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if len(gpus) > 0 else '/CPU:0'
        with tf.device(device):
            warped = tf.raw_ops.ImageProjectiveTransformV3(
                images=img_tf,
                transforms=transforms,
                output_shape=out_shape,
                interpolation="BILINEAR",
                fill_mode="REFLECT",
                fill_value=0.0
            )
        warped_np = (warped[0].numpy() * 255.0).astype(np.uint8)
        warped_bgr = cv2.cvtColor(warped_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("TF warp failed, falling back to cv2.warpAffine:", e)
        t1 = np.float32(t1_rect)
        t2 = np.float32(t2_rect)
        M = cv2.getAffineTransform(t1, t2)
        size = (w2, h2)
        warped_bgr = cv2.warpAffine(src_patch, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((h2, w2, 3), dtype=np.uint8)
    t2_rect_i = np.int32([t2_rect])
    cv2.fillConvexPoly(mask, t2_rect_i, (1,1,1), 16, 0)

    dst_region = dst_img[y2:y2+h2, x2:x2+w2]
    if dst_region.shape[0] != h2 or dst_region.shape[1] != w2:
        return
    mask_bool = (mask.astype(bool))
    combined = dst_region.copy()
    combined[mask_bool] = warped_bgr[mask_bool]
    dst_img[y2:y2+h2, x2:x2+w2] = combined

# CPU fallback warp
def warp_triangle_cv(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), cv2.LINE_AA)

    img1_rect = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if img1_rect.size == 0:
        return

    t1_rect_np = np.float32(t1_rect)
    t2_rect_np = np.float32(t2_rect)
    M = cv2.getAffineTransform(t1_rect_np, t2_rect_np)
    size = (r2[2], r2[3])
    warped = cv2.warpAffine(img1_rect, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    dst_region = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst_region = dst_region * (1 - mask) + warped * mask
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_region

# =============================
# face_swap
# =============================
def face_swap(src_bgr: np.ndarray, dst_bgr: np.ndarray,
              src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    src = src_bgr.copy()
    dst = dst_bgr.copy()
    rect = rect_from_points(dst_pts)
    tris = delaunay_triangulation(rect, dst_pts)
    print("Number of triangles:", len(tris))

    warped_src = dst.copy()
    use_tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0

    for (i, j, k) in tris:
        t_src = [src_pts[i], src_pts[j], src_pts[k]]
        t_dst = [dst_pts[i], dst_pts[j], dst_pts[k]]
        try:
            if use_tf_gpu:
                warp_triangle_tf(src, warped_src, t_src, t_dst)
            else:
                warp_triangle_cv(src, warped_src, t_src, t_dst)
        except Exception as e:
            print("warp error:", e)
            try:
                warp_triangle_cv(src, warped_src, t_src, t_dst)
            except Exception:
                pass

    hull = cv2.convexHull(dst_pts.astype(np.int32))
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    (x, y, w, h) = cv2.boundingRect(hull)
    center = (x + w // 2, y + h // 2)

    try:
        output = cv2.seamlessClone(np.uint8(warped_src), np.uint8(dst_bgr), mask, center, cv2.MIXED_CLONE)
    except Exception as e:
        print("seamlessClone failed:", e)
        output = dst_bgr
    return output

# =============================
# CameraWorker
# =============================
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, src_image: Optional[np.ndarray] = None, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        self.src_image = src_image
        self.detector = MediaPipeFaceMeshDetector(static_image_mode=False) if MP_AVAILABLE else None
        self.src_landmarks: Optional[np.ndarray] = None
        self.smoother = EMASmoother(alpha=0.6)
        self.show_live_landmarks: bool = False

    def set_source_image(self, img: Optional[np.ndarray]):
        self.src_image = img
        self.src_landmarks = None
        if img is not None and self.detector is not None:
            try:
                pts = self.detector.detect(img)
                if pts is not None:
                    self.src_landmarks = pts
                    print("Source landmarks computed:", pts.shape)
            except Exception as e:
                print("Error computing source landmarks:", e)

    def set_show_live_landmarks(self, value: bool):
        self.show_live_landmarks = bool(value)

    def run(self):
        self.running = True
        print("Opening camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Camera failed to open")
            self.running = False
            return

        if self.src_image is not None and self.detector is not None and self.src_landmarks is None:
            self.src_landmarks = self.detector.detect(self.src_image)
            print("Initial source landmarks:", None if self.src_landmarks is None else self.src_landmarks.shape)

        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                continue

            out = frame
            dst_landmarks_raw = None
            if self.detector is not None:
                dst_landmarks_raw = self.detector.detect(frame)
            dst_landmarks = self.smoother.update(dst_landmarks_raw)

            if self.src_image is not None and self.src_landmarks is not None and dst_landmarks is not None:
                try:
                    out = face_swap(self.src_image, frame, self.src_landmarks, dst_landmarks)
                except Exception as e:
                    print("face_swap error:", e)
                    out = frame

            if self.show_live_landmarks and dst_landmarks is not None:
                vis = out.copy()
                for (x, y) in dst_landmarks.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
                out = vis

            self.frame_ready.emit(out)

        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(2000)

# =============================
# MainWindow
# =============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Face Swap (Full 468 Mesh) - TF GPU Warp Enabled")
        self.resize(1100, 720)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background:#111; color:#ddd;")

        self.btn_browse = QtWidgets.QPushButton("Browse Source Photo…")
        self.btn_start = QtWidgets.QPushButton("Start Camera")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.chk_preview_src = QtWidgets.QCheckBox("Preview Source Landmarks")
        self.show_live_landmarks_checkbox = QtWidgets.QCheckBox("Show Live Landmarks")

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btn_browse)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addStretch(1)
        controls.addWidget(self.chk_preview_src)
        controls.addWidget(self.show_live_landmarks_checkbox)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addWidget(self.video_label, 1)
        v.addLayout(controls)
        self.setCentralWidget(central)

        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.show_live_landmarks_checkbox.toggled.connect(self.on_toggle_live_landmarks)

        self.worker: Optional[CameraWorker] = None
        self.src_image: Optional[np.ndarray] = None

        if not MP_AVAILABLE:
            self.status.showMessage("MediaPipe bulunamadı — 'pip install mediapipe' yapın.", 10000)

    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Source Face", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status.showMessage("Failed to load image.", 5000)
            return
        self.src_image = img
        self.status.showMessage(f"Loaded: {os.path.basename(path)}", 5000)
        if self.chk_preview_src.isChecked() and MP_AVAILABLE:
            det = MediaPipeFaceMeshDetector(static_image_mode=True)
            pts = det.detect(img)
            preview = img.copy()
            if pts is not None:
                for (x, y) in pts.astype(int):
                    cv2.circle(preview, (int(x), int(y)), 1, (0,255,0), -1)
            qimg = to_qimage(preview)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        if self.worker is not None:
            self.worker.set_source_image(img)

    def on_start(self):
        if self.worker is not None:
            self.worker.stop()
        self.worker = CameraWorker(src_image=self.src_image)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.start()

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def on_toggle_live_landmarks(self, state: bool):
        if self.worker is not None:
            self.worker.set_show_live_landmarks(state)

    def on_frame(self, frame: np.ndarray):
        qimg = to_qimage(frame)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

# =============================
# main
# =============================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
PyQt6 + OpenCV face-swap with full 468-point MediaPipe landmarks.
GPU/CUDA acceleration for full-face warping using TensorFlow.
SeamlessClone used for final blending.
Requires: pip install pyqt6 opencv-python mediapipe numpy tensorflow
"""
from __future__ import annotations
import sys, os
from typing import Optional, Tuple, List

import numpy as np
import cv2
import tensorflow as tf
print("TF version:", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

# --- TF device ---
TF_GPUS = tf.config.list_physical_devices('GPU')
TF_DEVICE = '/GPU:0' if len(TF_GPUS) > 0 else '/CPU:0'
print("TF device selected:", TF_DEVICE)

# --- MediaPipe ---
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("MediaPipe version:", mp.__version__)
except Exception as e:
    MP_AVAILABLE = False
    print("MediaPipe import error:", e)

# --- PyQt6 ---
from PyQt6 import QtCore, QtGui, QtWidgets

# =============================
# Utilities
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
# TF GPU warp
# =============================
def tf_warp_image_with_homography(src_bgr: np.ndarray, H_inv: np.ndarray, out_size: Tuple[int,int]):
    h_out, w_out = out_size[1], out_size[0]
    H = H_inv.astype(np.float32)
    a0,a1,a2 = float(H[0,0]), float(H[0,1]), float(H[0,2])
    a3,a4,a5 = float(H[1,0]), float(H[1,1]), float(H[1,2])
    a6,a7 = float(H[2,0]), float(H[2,1])
    transform = np.array([a0,a1,a2,a3,a4,a5,a6,a7], dtype=np.float32).reshape(1,8)
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    src_tf = np.expand_dims(src_rgb, axis=0)
    try:
        with tf.device(TF_DEVICE):
            tf_src = tf.convert_to_tensor(src_tf, dtype=tf.float32)
            tf_transforms = tf.convert_to_tensor(transform, dtype=tf.float32)
            out_shape = tf.constant([h_out, w_out], dtype=tf.int32)
            warped = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf_src,
                transforms=tf_transforms,
                output_shape=out_shape,
                interpolation="BILINEAR",
                fill_mode="REFLECT",
                fill_value=0.0
            )
            warped_np = (warped[0].numpy()*255.0).astype(np.uint8)
            warped_bgr = cv2.cvtColor(warped_np, cv2.COLOR_RGB2BGR)
            return warped_bgr
    except Exception as e:
        print("TF warp failed, fallback cv2.warpPerspective:", e)
        return cv2.warpPerspective(src_bgr, np.linalg.inv(H_inv), (w_out,h_out),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# =============================
# Face swap (GPU warp + seamlessClone)
# =============================
def pyramid_blend(src, dst, mask, num_levels=4):
    """Multi-band (pyramid) blending fallback."""
    import cv2
    import numpy as np

    mask_f = (mask.astype("float32") / 255.0)[..., None]

    gp_mask = [mask_f]
    gp_src = [src.astype("float32")]
    gp_dst = [dst.astype("float32")]
    for i in range(num_levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))
        gp_src.append(cv2.pyrDown(gp_src[-1]))
        gp_dst.append(cv2.pyrDown(gp_dst[-1]))

    lp_src = []
    lp_dst = []
    for i in range(num_levels):
        sz = (gp_src[i].shape[1], gp_src[i].shape[0])
        GE = cv2.pyrUp(gp_src[i+1], dstsize=sz)
        L = gp_src[i] - GE
        lp_src.append(L)

        GE2 = cv2.pyrUp(gp_dst[i+1], dstsize=sz)
        L2 = gp_dst[i] - GE2
        lp_dst.append(L2)

    lp_src.append(gp_src[-1])
    lp_dst.append(gp_dst[-1])

    lp_res = []
    for ls, ld, gm in zip(lp_src, lp_dst, gp_mask):
        lp_res.append(ls * gm + ld * (1.0 - gm))

    res = lp_res[-1]
    for i in range(num_levels-1, -1, -1):
        sz = (lp_res[i].shape[1], lp_res[i].shape[0])
        res = cv2.pyrUp(res, dstsize=sz) + lp_res[i]

    return np.clip(res, 0, 255).astype("uint8")


def improved_color_transfer(src_region, dst_img, mask_region):
    """LAB tabanlı renk eşleme + CLAHE (sadece mask içinde)."""
    import cv2, numpy as np

    ys, xs = np.where(mask_region == 255)
    if len(xs) == 0:
        return src_region
    x0, x1 = np.min(xs), np.max(xs)
    y0, y1 = np.min(ys), np.max(ys)

    src_patch = src_region[y0:y1+1, x0:x1+1]
    dst_patch = dst_img[y0:y1+1, x0:x1+1]
    mask_patch = mask_region[y0:y1+1, x0:x1+1]

    src_lab = cv2.cvtColor(src_patch, cv2.COLOR_BGR2LAB).astype("float32")
    dst_lab = cv2.cvtColor(dst_patch, cv2.COLOR_BGR2LAB).astype("float32")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    src_lab[:, :, 0] = clahe.apply(src_lab[:, :, 0].astype("uint8")).astype("float32")

    eps = 1e-6
    for c in range(3):
        s_vals = src_lab[:, :, c][mask_patch == 255]
        d_vals = dst_lab[:, :, c][mask_patch == 255]
        if s_vals.size == 0 or d_vals.size == 0:
            continue
        s_mean, s_std = s_vals.mean(), s_vals.std()
        d_mean, d_std = d_vals.mean(), d_vals.std()
        src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (d_std / (s_std + eps)) + d_mean

    src_lab = np.clip(src_lab, 0, 255).astype("uint8")
    src_corr = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

    out = src_region.copy()
    out[y0:y1+1, x0:x1+1][mask_patch == 255] = src_corr[mask_patch == 255]
    return out


def face_swap(src_bgr: np.ndarray, dst_bgr: np.ndarray,
              src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Delaunay triangle-based face swap + gelişmiş renk eşleme + maske yumuşatma.
    Optimize edilmiş versiyon: mask3 repeat() kaldırıldı, cv2.bitwise_and + cv2.add kullanıldı.
    """
    try:
        h_dst, w_dst = dst_bgr.shape[:2]
        src_pts_f = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
        dst_pts_f = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)

        hull_index = cv2.convexHull(dst_pts_f.astype(np.int32), returnPoints=False).flatten()
        if hull_index.size == 0:
            print("Empty hull_index")
            return dst_bgr

        dst_hull = dst_pts_f[hull_index]
        src_hull = src_pts_f[hull_index]

        rect = (0, 0, w_dst, h_dst)
        subdiv = cv2.Subdiv2D(rect)
        for (x, y) in dst_pts_f:
            subdiv.insert((float(x), float(y)))
        triangleList = subdiv.getTriangleList()

        def find_index(pt):
            x, y = pt
            dists = np.linalg.norm(dst_pts_f - np.array([x, y]), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 2.0:
                return -1
            return idx

        warped_src = np.zeros_like(dst_bgr)
        mask_acc = np.zeros((h_dst, w_dst), dtype=np.uint8)

        for t in triangleList:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            idx0 = find_index(pts[0])
            idx1 = find_index(pts[1])
            idx2 = find_index(pts[2])
            if idx0 == -1 or idx1 == -1 or idx2 == -1:
                continue

            t_dst = np.array([dst_pts_f[idx0], dst_pts_f[idx1], dst_pts_f[idx2]], dtype=np.float32)
            t_src = np.array([src_pts_f[idx0], src_pts_f[idx1], src_pts_f[idx2]], dtype=np.float32)

            r_dst = cv2.boundingRect(t_dst)
            r_src = cv2.boundingRect(t_src)
            x_dst, y_dst, w_t_dst, h_t_dst = r_dst
            x_src, y_src, w_t_src, h_t_src = r_src
            if w_t_dst == 0 or h_t_dst == 0 or w_t_src == 0 or h_t_src == 0:
                continue

            t_dst_offset = np.array([[p[0]-x_dst, p[1]-y_dst] for p in t_dst], dtype=np.float32)
            t_src_offset = np.array([[p[0]-x_src, p[1]-y_src] for p in t_src], dtype=np.float32)

            src_patch = src_bgr[y_src:y_src+h_t_src, x_src:x_src+w_t_src]
            if src_patch.size == 0:
                continue

            M = cv2.getAffineTransform(t_src_offset, t_dst_offset)
            warped_patch = cv2.warpAffine(src_patch, M, (w_t_dst, h_t_dst),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask = np.zeros((h_t_dst, w_t_dst), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(t_dst_offset), 255)

            roi = warped_src[y_dst:y_dst+h_t_dst, x_dst:x_dst+w_t_dst]
            roi_mask = mask_acc[y_dst:y_dst+h_t_dst, x_dst:x_dst+w_t_dst]
            roi[mask == 255] = warped_patch[mask == 255]
            roi_mask[mask == 255] = 255

        if np.count_nonzero(mask_acc) < 50:
            if len(hull_index) >= 3:
                M, inliers = cv2.estimateAffinePartial2D(src_hull, dst_hull)
                if M is not None:
                    H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
                    H_inv = np.linalg.inv(H)
                    warped_src = tf_warp_image_with_homography(src_bgr, H_inv, (w_dst, h_dst))
                    mask_acc = np.zeros((h_dst, w_dst), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_acc, np.int32(dst_hull), 255)
                else:
                    return dst_bgr
            else:
                return dst_bgr

        # --- maske yumuşatma ---
        face_area = max(1, int(np.sqrt(w_dst * h_dst) / 50))
        k_erode = max(1, face_area // 2)
        k_dilate = max(1, face_area)
        kernel_e = np.ones((k_erode, k_erode), np.uint8)
        kernel_d = np.ones((k_dilate, k_dilate), np.uint8)
        mask_bin = cv2.erode(mask_acc, kernel_e, iterations=1)
        mask_bin = cv2.dilate(mask_bin, kernel_d, iterations=1)

        sigma = max(3, face_area * 1.5)
        mask_blur = cv2.GaussianBlur(mask_bin, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask_blur = cv2.normalize(mask_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- renk eşleme sadece mask içinde ---
        warped_region = cv2.bitwise_and(warped_src, warped_src, mask=mask_acc)
        warped_region = improved_color_transfer(warped_region, dst_bgr, mask_acc)

        inv_mask = cv2.bitwise_not(mask_acc)
        base_bg = cv2.bitwise_and(dst_bgr, dst_bgr, mask=inv_mask)
        warped_src_full = cv2.add(base_bg, warped_region)

        # --- seamlessClone dene ---
        hull_rect = cv2.boundingRect(np.int32(dst_hull))
        cx, cy, cw, ch = hull_rect
        center = (cx + cw // 2, cy + ch // 2)
        center = (np.clip(center[0], 0, w_dst-1), np.clip(center[1], 0, h_dst-1))

        _, mask_blur = cv2.threshold(mask_blur, 10, 255, cv2.THRESH_BINARY)
        mask_blur = mask_blur.astype(np.uint8)

        output = None
        try:
            output = cv2.seamlessClone(warped_src_full, dst_bgr, mask_blur, center, cv2.MIXED_CLONE)
        except Exception as e1:
            try:
                output = cv2.seamlessClone(warped_src_full, dst_bgr, mask_blur, center, cv2.NORMAL_CLONE)
            except Exception as e2:
                print("seamlessClone hata:", e1, e2)
                output = None

        if output is None:
            output = pyramid_blend(warped_src_full, dst_bgr, mask_blur, num_levels=4)

        return output

    except Exception as ee:
        print("face_swap failed:", ee)
        return dst_bgr

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
            pts = self.detector.detect(img)
            if pts is not None:
                self.src_landmarks = pts
                print("Source landmarks computed:", pts.shape)
    def set_show_live_landmarks(self, value: bool):
        self.show_live_landmarks = bool(value)
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Camera failed to open")
            self.running = False
            return
        if self.src_image is not None and self.detector is not None and self.src_landmarks is None:
            self.src_landmarks = self.detector.detect(self.src_image)
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok: continue
            out = frame
            dst_landmarks_raw = None
            if self.detector is not None:
                dst_landmarks_raw = self.detector.detect(frame)
            dst_landmarks = self.smoother.update(dst_landmarks_raw)
            if self.src_image is not None and self.src_landmarks is not None and dst_landmarks is not None:
                out = face_swap(self.src_image, frame, self.src_landmarks, dst_landmarks)
            if self.show_live_landmarks and dst_landmarks is not None:
                vis = out.copy()
                for (x,y) in dst_landmarks.astype(int):
                    cv2.circle(vis,(int(x),int(y)),1,(0,255,0),-1)
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
        self.resize(1100,720)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640,360)
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
        v.addWidget(self.video_label,1)
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
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Source Face","","Images (*.png *.jpg *.jpeg)")
        if not path: return
        img = cv2.imread(path)
        if img is None:
            self.status.showMessage("Failed to load image.",5000)
            return
        self.src_image = img
        self.status.showMessage(f"Loaded: {os.path.basename(path)}",5000)
        if self.chk_preview_src.isChecked() and MP_AVAILABLE:
            det = MediaPipeFaceMeshDetector(static_image_mode=True)
            pts = det.detect(img)
            preview = img.copy()
            if pts is not None:
                for (x,y) in pts.astype(int):
                    cv2.circle(preview,(int(x),int(y)),1,(0,255,0),-1)
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
    def on_toggle_live_landmarks(self,state: bool):
        if self.worker is not None:
            self.worker.set_show_live_landmarks(state)
    def on_frame(self, frame: np.ndarray):
        qimg = to_qimage(frame)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

# =============================
# Main
# =============================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

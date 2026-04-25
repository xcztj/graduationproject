# -*- coding: utf-8 -*-
"""
VGA-Net 视网膜血管分割 GUI (PyQt5)
功能：
  1. 加载眼底图像（支持 .tif / .png / .jpg / .bmp）
  2. 调用模型进行血管分割
  3. 显示原图、分割结果、叠加对比图
  4. 支持保存分割结果

运行方式：
    cd /root/autodl-tmp/VGA-Net/GUI
    python vga_net_gui.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from inference import VGAInferencer


class SegmentationThread(QThread):
    """后台推理线程，防止 GUI 卡死"""
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # rgb, prob, binary
    error = pyqtSignal(str)

    def __init__(self, inferencer, image_path):
        super().__init__()
        self.inferencer = inferencer
        self.image_path = image_path

    def run(self):
        try:
            img_rgb, pred_prob, pred_binary = self.inferencer.predict(self.image_path)
            self.finished.emit(img_rgb, pred_prob, pred_binary)
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """支持缩放的图像显示标签"""
    def __init__(self, title=""):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
        self.setText(title if title else "未加载图像")
        self._pixmap = None
        self._raw_image = None

    def set_image(self, img_array):
        """设置 numpy 图像 (H, W, C) 或 (H, W)"""
        self._raw_image = img_array.copy()
        self.update_pixmap()

    def update_pixmap(self):
        if self._raw_image is None:
            return
        h, w = self._raw_image.shape[:2]
        if len(self._raw_image.shape) == 3:
            # RGB
            bytes_per_line = 3 * w
            qimg = QImage(self._raw_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Gray
            qimg = QImage(self._raw_image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmap = pixmap
        self._scale_and_set()

    def _scale_and_set(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_and_set()

    def save_image(self, path):
        """保存当前显示的原始图像"""
        if self._raw_image is not None:
            if len(self._raw_image.shape) == 3:
                cv2.imwrite(path, cv2.cvtColor(self._raw_image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(path, self._raw_image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGA-Net 视网膜血管分割系统")
        self.setGeometry(100, 100, 1400, 900)

        # 模型路径（可修改）
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'Train', 'best_model.pt')
        self.inferencer = None
        self.current_image_path = None
        self.seg_thread = None

        self._setup_ui()
        self._load_model()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # === 顶部按钮栏 ===
        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("📂 选择图像")
        self.btn_load.setToolTip("选择一张眼底图像 (.tif / .png / .jpg)")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_image)

        self.btn_segment = QPushButton("🔬 运行分割")
        self.btn_segment.setToolTip("调用 VGA-Net 模型进行血管分割")
        self.btn_segment.setMinimumHeight(40)
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.run_segmentation)

        self.btn_save = QPushButton("💾 保存结果")
        self.btn_save.setToolTip("保存分割结果图")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_result)

        self.btn_save_compare = QPushButton("💾 保存对比图")
        self.btn_save_compare.setToolTip("保存原图+分割结果的叠加对比图")
        self.btn_save_compare.setMinimumHeight(40)
        self.btn_save_compare.setEnabled(False)
        self.btn_save_compare.clicked.connect(self.save_compare)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_segment)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_save_compare)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout)

        # === 进度条 ===
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # === 中部图像显示区 ===
        splitter = QSplitter(Qt.Horizontal)

        # 原始图像
        group_orig = QGroupBox("原始图像")
        orig_layout = QVBoxLayout(group_orig)
        self.label_orig = ImageLabel("请点击「选择图像」加载眼底图像")
        orig_layout.addWidget(self.label_orig)
        splitter.addWidget(group_orig)

        # 分割结果
        group_seg = QGroupBox("分割结果")
        seg_layout = QVBoxLayout(group_seg)
        self.label_seg = ImageLabel("分割结果将显示在这里")
        seg_layout.addWidget(self.label_seg)
        splitter.addWidget(group_seg)

        # 叠加对比
        group_overlay = QGroupBox("叠加对比")
        overlay_layout = QVBoxLayout(group_overlay)
        self.label_overlay = ImageLabel("原图与分割结果的叠加对比")
        overlay_layout.addWidget(self.label_overlay)
        splitter.addWidget(group_overlay)

        splitter.setSizes([450, 450, 450])
        main_layout.addWidget(splitter, stretch=1)

        # === 底部信息栏 ===
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setPlaceholderText("运行日志...")
        main_layout.addWidget(self.info_text)

        # 状态栏
        self.statusBar().showMessage(f"模型路径: {self.model_path}")

    def _log(self, msg):
        self.info_text.append(msg)

    def _load_model(self):
        """加载分割模型"""
        try:
            if not os.path.exists(self.model_path):
                self._log(f"⚠️ 模型文件不存在: {self.model_path}")
                self.statusBar().showMessage("模型未加载，请检查路径")
                return
            self._log("正在加载模型...")
            self.inferencer = VGAInferencer(self.model_path)
            self._log(f"✅ 模型加载成功: {os.path.basename(self.model_path)}")
            self.statusBar().showMessage("模型已加载，请选择图像")
        except Exception as e:
            self._log(f"❌ 模型加载失败: {e}")
            self.statusBar().showMessage("模型加载失败")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    def load_image(self):
        """选择图像文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择眼底图像", "",
            "图像文件 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return
        self.current_image_path = path
        self._log(f"已加载图像: {os.path.basename(path)}")

        # 读取并显示原图
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "警告", "无法读取图像文件")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.label_orig.set_image(img_rgb)

        # 清空之前的结果
        self.label_seg.setText("请点击「运行分割」")
        self.label_seg._raw_image = None
        self.label_seg._pixmap = None
        self.label_overlay.setText("等待分割结果...")
        self.label_overlay._raw_image = None
        self.label_overlay._pixmap = None

        self.btn_segment.setEnabled(self.inferencer is not None)
        self.btn_save.setEnabled(False)
        self.btn_save_compare.setEnabled(False)
        self.statusBar().showMessage(f"已加载: {os.path.basename(path)}  |  尺寸: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    def run_segmentation(self):
        """运行分割推理"""
        if self.current_image_path is None or self.inferencer is None:
            return

        self.btn_segment.setEnabled(False)
        self.progress.setVisible(True)
        self._log("🔬 正在分割，请稍候...")

        self.seg_thread = SegmentationThread(self.inferencer, self.current_image_path)
        self.seg_thread.finished.connect(self._on_segmentation_finished)
        self.seg_thread.error.connect(self._on_segmentation_error)
        self.seg_thread.start()

    def _on_segmentation_finished(self, img_rgb, pred_prob, pred_binary):
        self.progress.setVisible(False)

        # 显示二值分割结果 (白色血管，黑色背景)
        seg_display = (pred_binary * 255).astype(np.uint8)
        self.label_seg.set_image(seg_display)

        # 生成叠加对比图：原图 + 红色血管叠加
        overlay = img_rgb.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[pred_binary == 1] = [255, 0, 0]  # 红色
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        self.label_overlay.set_image(overlay)

        # 保存当前结果用于后续保存
        self._last_prob = pred_prob
        self._last_binary = pred_binary
        self._last_overlay = overlay

        self.btn_segment.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_save_compare.setEnabled(True)

        vessel_ratio = pred_binary.mean() * 100
        self._log(f"✅ 分割完成！血管像素占比: {vessel_ratio:.2f}%")
        self.statusBar().showMessage(f"分割完成  |  血管占比: {vessel_ratio:.2f}%")

    def _on_segmentation_error(self, error_msg):
        self.progress.setVisible(False)
        self.btn_segment.setEnabled(True)
        self._log(f"❌ 分割失败: {error_msg}")
        QMessageBox.critical(self, "分割错误", error_msg)

    def save_result(self):
        """保存分割二值图"""
        if self._last_binary is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存分割结果", "segmentation_result.png",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg);;所有文件 (*.*)"
        )
        if path:
            seg_img = (self._last_binary * 255).astype(np.uint8)
            cv2.imwrite(path, seg_img)
            self._log(f"💾 分割结果已保存: {path}")

    def save_compare(self):
        """保存叠加对比图"""
        if self._last_overlay is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存对比图", "comparison_overlay.png",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg);;所有文件 (*.*)"
        )
        if path:
            cv2.imwrite(path, cv2.cvtColor(self._last_overlay, cv2.COLOR_RGB2BGR))
            self._log(f"💾 对比图已保存: {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #fafafa;
        }
        QPushButton {
            background-color: #4a90d9;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #357abd;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 4px;
            background-color: #ffffff;
            font-family: Consolas, Monaco, monospace;
            font-size: 12px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

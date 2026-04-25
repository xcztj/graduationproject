# -*- coding: utf-8 -*-
"""
VGA-Net 可视化推理界面 (PyQt5)
功能：
  1. 加载模型权重
  2. 选择单张图片或批量处理
  3. 显示原图 / Ground Truth / 预测结果
  4. 显示各项指标 (Dice, Accuracy, SE, SP, MCC)
  5. 保存预测结果
"""

import sys
import os

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QGridLayout, QProgressBar, QTextEdit, QSplitter, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from inference import VGAInferencer


def numpy_to_qpixmap(img_array):
    """将 numpy 数组转换为 QPixmap"""
    if len(img_array.shape) == 2:
        # 灰度图
        height, width = img_array.shape
        bytes_per_line = width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        # RGB 图
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        # RGB -> BGR for Qt
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)


class InferenceThread(QThread):
    """后台推理线程"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, inferencer, image_path, mask_path=None):
        super().__init__()
        self.inferencer = inferencer
        self.image_path = image_path
        self.mask_path = mask_path
    
    def run(self):
        try:
            img_rgb, pred_prob, pred_binary = self.inferencer.predict(self.image_path)
            
            mask = None
            if self.mask_path and os.path.exists(self.mask_path):
                mask = self.inferencer.load_mask(self.mask_path, img_rgb.shape)
            
            metrics = self.inferencer.compute_metrics(pred_binary, mask)
            
            self.finished.emit({
                'image': img_rgb,
                'pred_prob': pred_prob,
                'pred_binary': pred_binary,
                'mask': mask,
                'metrics': metrics
            })
        except Exception as e:
            self.error.emit(str(e))


class BatchInferenceThread(QThread):
    """批量推理线程"""
    progress = pyqtSignal(int, int)
    image_ready = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, inferencer, image_dir, mask_dir=None):
        super().__init__()
        self.inferencer = inferencer
        self.image_dir = image_dir
        self.mask_dir = mask_dir
    
    def run(self):
        try:
            import glob
            image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
            total = len(image_paths)
            
            for i, img_path in enumerate(image_paths):
                basename = os.path.splitext(os.path.basename(img_path))[0]
                img_rgb, pred_prob, pred_binary = self.inferencer.predict(img_path)
                
                mask = None
                if self.mask_dir and os.path.exists(self.mask_dir):
                    mask_path = os.path.join(self.mask_dir, f"{basename}_manual1.gif")
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(self.mask_dir, f"{basename}_mask.gif")
                    if os.path.exists(mask_path):
                        mask = self.inferencer.load_mask(mask_path, img_rgb.shape)
                
                metrics = self.inferencer.compute_metrics(pred_binary, mask)
                
                self.image_ready.emit({
                    'name': basename,
                    'image': img_rgb,
                    'pred_prob': pred_prob,
                    'pred_binary': pred_binary,
                    'mask': mask,
                    'metrics': metrics
                })
                self.progress.emit(i + 1, total)
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGA-Net 视网膜血管分割可视化系统")
        self.setGeometry(100, 100, 1400, 900)
        
        self.inferencer = None
        self.current_result = None
        self.batch_results = []
        
        self.init_ui()
    
    def init_ui(self):
        # 主窗口中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局：左右分割
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ========== 左侧控制面板 ==========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # --- 模型加载区域 ---
        model_group = QGroupBox("模型加载")
        model_layout = QVBoxLayout(model_group)
        
        self.model_path_label = QLabel("未加载模型")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self.model_path_label)
        
        btn_load_model = QPushButton("加载模型权重")
        btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(btn_load_model)
        
        left_layout.addWidget(model_group)
        
        # --- 图片选择区域 ---
        image_group = QGroupBox("数据选择")
        image_layout = QVBoxLayout(image_group)
        
        btn_load_image = QPushButton("选择单张图片")
        btn_load_image.clicked.connect(self.load_single_image)
        image_layout.addWidget(btn_load_image)
        
        btn_load_batch = QPushButton("选择图片文件夹（批量）")
        btn_load_batch.clicked.connect(self.load_batch_images)
        image_layout.addWidget(btn_load_batch)
        
        self.image_path_label = QLabel("未选择图片")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setStyleSheet("color: gray;")
        image_layout.addWidget(self.image_path_label)
        
        left_layout.addWidget(image_group)
        
        # --- 运行按钮 ---
        btn_run = QPushButton("运行推理")
        btn_run.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        btn_run.clicked.connect(self.run_inference)
        left_layout.addWidget(btn_run)
        
        # --- 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # --- 指标显示区域 ---
        metrics_group = QGroupBox("评估指标")
        metrics_layout = QGridLayout(metrics_group)
        
        self.metric_labels = {}
        metrics = ['Accuracy', 'Dice', 'SE', 'SP', 'MCC']
        for i, name in enumerate(metrics):
            label_name = QLabel(f"{name}:")
            label_value = QLabel("-")
            label_value.setStyleSheet("color: blue; font-weight: bold;")
            metrics_layout.addWidget(label_name, i, 0)
            metrics_layout.addWidget(label_value, i, 1)
            self.metric_labels[name] = label_value
        
        left_layout.addWidget(metrics_group)
        
        # --- 保存按钮 ---
        btn_save = QPushButton("保存当前预测结果")
        btn_save.clicked.connect(self.save_result)
        left_layout.addWidget(btn_save)
        
        # 添加弹簧
        left_layout.addStretch()
        
        # ========== 右侧图像显示区域 ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像显示网格
        image_grid = QWidget()
        grid_layout = QHBoxLayout(image_grid)
        
        # 原图
        self.original_group = self.create_image_group("原始图像")
        grid_layout.addWidget(self.original_group)
        
        # Ground Truth
        self.gt_group = self.create_image_group("Ground Truth")
        grid_layout.addWidget(self.gt_group)
        
        # 预测结果
        self.pred_group = self.create_image_group("预测结果")
        grid_layout.addWidget(self.pred_group)
        
        right_layout.addWidget(image_grid)
        
        # 批量结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['文件名', 'Accuracy', 'Dice', 'SE', 'SP', 'MCC'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMaximumHeight(250)
        self.table.itemClicked.connect(self.on_table_item_clicked)
        right_layout.addWidget(self.table)
        
        # 添加面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])
        
        # 状态栏
        self.statusBar().showMessage("就绪 | 请先加载模型权重")
    
    def create_image_group(self, title):
        """创建图像显示组"""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        
        label = QLabel("未加载")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(400, 400)
        label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #cccccc;
                color: #999999;
            }
        """)
        layout.addWidget(label)
        
        return group
    
    def get_image_label(self, group):
        """获取组内的图像标签"""
        return group.layout().itemAt(0).widget()
    
    def set_image(self, group, img_array, scale=380):
        """设置组内的图像"""
        label = self.get_image_label(group)
        pixmap = numpy_to_qpixmap(img_array)
        scaled_pixmap = pixmap.scaled(scale, scale, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setStyleSheet("border: 1px solid #cccccc;")
    
    def load_model(self):
        """加载模型权重"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "/root/autodl-tmp/VGA-Net/Train",
            "PyTorch模型 (*.pt *.pth)"
        )
        if path:
            try:
                self.inferencer = VGAInferencer(path)
                self.model_path_label.setText(os.path.basename(path))
                self.model_path_label.setStyleSheet("color: green;")
                self.statusBar().showMessage(f"模型加载成功: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
    
    def load_single_image(self):
        """选择单张图片"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片",
            "/root/autodl-tmp/VGA-Net/DRIVE/test/images",
            "图像文件 (*.tif *.png *.jpg *.bmp)"
        )
        if path:
            self.current_image_path = path
            self.current_mask_path = self.find_mask_path(path)
            self.image_path_label.setText(os.path.basename(path))
            self.image_path_label.setStyleSheet("color: green;")
            self.batch_mode = False
            self.statusBar().showMessage(f"已选择图片: {os.path.basename(path)}")
    
    def load_batch_images(self):
        """选择图片文件夹"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹",
            "/root/autodl-tmp/VGA-Net/DRIVE/test"
        )
        if dir_path:
            self.current_image_dir = dir_path
            self.current_mask_dir = os.path.join(os.path.dirname(dir_path), '1st_manual')
            if not os.path.exists(self.current_mask_dir):
                self.current_mask_dir = os.path.join(os.path.dirname(dir_path), 'mask')
            
            self.image_path_label.setText(f"文件夹: {os.path.basename(dir_path)}")
            self.image_path_label.setStyleSheet("color: green;")
            self.batch_mode = True
            self.statusBar().showMessage(f"已选择文件夹: {dir_path}")
    
    def find_mask_path(self, image_path):
        """根据图片路径查找对应的 mask"""
        dir_name = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 尝试多种 mask 命名格式
        possible_masks = [
            os.path.join(dir_name.replace('images', '1st_manual'), f"{base_name.replace('test', 'manual1')}.gif"),
            os.path.join(dir_name.replace('images', '1st_manual'), f"{base_name.replace('training', 'manual1')}.gif"),
            os.path.join(dir_name.replace('images', 'mask'), f"{base_name}_mask.gif"),
            os.path.join(dir_name.replace('images', 'mask'), f"{base_name}.gif"),
        ]
        
        for mask_path in possible_masks:
            if os.path.exists(mask_path):
                return mask_path
        return None
    
    def run_inference(self):
        """运行推理"""
        if self.inferencer is None:
            QMessageBox.warning(self, "警告", "请先加载模型权重！")
            return
        
        if not hasattr(self, 'batch_mode'):
            QMessageBox.warning(self, "警告", "请先选择图片或文件夹！")
            return
        
        if self.batch_mode:
            self.run_batch_inference()
        else:
            self.run_single_inference()
    
    def run_single_inference(self):
        """单张推理"""
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return
        
        self.thread = InferenceThread(
            self.inferencer,
            self.current_image_path,
            self.current_mask_path if hasattr(self, 'current_mask_path') else None
        )
        self.thread.finished.connect(self.on_single_inference_finished)
        self.thread.error.connect(self.on_inference_error)
        self.thread.start()
        
        self.statusBar().showMessage("正在推理...")
    
    def run_batch_inference(self):
        """批量推理"""
        if not hasattr(self, 'current_image_dir'):
            QMessageBox.warning(self, "警告", "请先选择图片文件夹！")
            return
        
        self.batch_results = []
        self.table.setRowCount(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.thread = BatchInferenceThread(
            self.inferencer,
            self.current_image_dir,
            self.current_mask_dir if hasattr(self, 'current_mask_dir') else None
        )
        self.thread.progress.connect(self.on_batch_progress)
        self.thread.image_ready.connect(self.on_batch_image_ready)
        self.thread.finished.connect(self.on_batch_finished)
        self.thread.error.connect(self.on_inference_error)
        self.thread.start()
        
        self.statusBar().showMessage("正在批量推理...")
    
    def on_single_inference_finished(self, result):
        """单张推理完成回调"""
        self.current_result = result
        
        # 显示图像
        self.set_image(self.original_group, result['image'])
        if result['mask'] is not None:
            self.set_image(self.gt_group, (result['mask'] * 255).astype(np.uint8))
        else:
            self.get_image_label(self.gt_group).setText("无 Ground Truth")
        self.set_image(self.pred_group, (result['pred_binary'] * 255).astype(np.uint8))
        
        # 显示指标
        self.display_metrics(result['metrics'])
        
        self.statusBar().showMessage("推理完成！")
    
    def on_batch_progress(self, current, total):
        """批量推理进度回调"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"正在处理: {current}/{total}")
    
    def on_batch_image_ready(self, result):
        """批量推理单张完成回调"""
        self.batch_results.append(result)
        
        # 添加到表格
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(result['name']))
        
        if result['metrics']:
            for i, key in enumerate(['Accuracy', 'Dice', 'SE', 'SP', 'MCC']):
                val = result['metrics'].get(key, 0)
                self.table.setItem(row, i + 1, QTableWidgetItem(f"{val:.4f}"))
        
        # 显示最后一张的结果
        self.current_result = result
        self.set_image(self.original_group, result['image'])
        if result['mask'] is not None:
            self.set_image(self.gt_group, (result['mask'] * 255).astype(np.uint8))
        else:
            self.get_image_label(self.gt_group).setText("无 Ground Truth")
        self.set_image(self.pred_group, (result['pred_binary'] * 255).astype(np.uint8))
        
        if result['metrics']:
            self.display_metrics(result['metrics'])
    
    def on_batch_finished(self):
        """批量推理完成回调"""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"批量推理完成！共处理 {len(self.batch_results)} 张图片")
        QMessageBox.information(self, "完成", f"批量推理完成！\n共处理 {len(self.batch_results)} 张图片")
    
    def on_inference_error(self, error_msg):
        """推理错误回调"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "推理错误", error_msg)
        self.statusBar().showMessage("推理失败")
    
    def on_table_item_clicked(self, item):
        """点击表格行显示对应结果"""
        row = item.row()
        if row < len(self.batch_results):
            result = self.batch_results[row]
            self.current_result = result
            self.set_image(self.original_group, result['image'])
            if result['mask'] is not None:
                self.set_image(self.gt_group, (result['mask'] * 255).astype(np.uint8))
            self.set_image(self.pred_group, (result['pred_binary'] * 255).astype(np.uint8))
            if result['metrics']:
                self.display_metrics(result['metrics'])
    
    def display_metrics(self, metrics):
        """显示指标"""
        if metrics is None:
            for label in self.metric_labels.values():
                label.setText("-")
            return
        
        for name, label in self.metric_labels.items():
            val = metrics.get(name, 0)
            label.setText(f"{val:.4f}")
    
    def save_result(self):
        """保存当前预测结果"""
        if self.current_result is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果！")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "保存预测结果",
            "/root/autodl-tmp/VGA-Net/results/prediction.png",
            "PNG图像 (*.png);;所有文件 (*.*)"
        )
        if path:
            pred = (self.current_result['pred_binary'] * 255).astype(np.uint8)
            cv2.imwrite(path, pred)
            self.statusBar().showMessage(f"结果已保存: {path}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

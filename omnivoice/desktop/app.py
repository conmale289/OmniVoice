#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OmniVoice Desktop application.

Provides a desktop-first UI for:
1) Single synthesis (voice clone / voice design / auto voice).
2) Batch inference from JSONL.
3) Local audio playback and export.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Callable, Optional

import soundfile as sf
import torch
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WorkerSignals(QObject):
    started = Signal()
    finished = Signal(object)
    failed = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        self.signals.started.emit()
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


@dataclass
class AudioRecord:
    created_at: datetime
    mode: str
    text_preview: str
    wav_path: str
    sample_rate: int
    duration_sec: float


class OmniVoiceDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniVoice Studio")
        self.setMinimumSize(1280, 820)
        self.thread_pool = QThreadPool(self)
        self.model: Optional[OmniVoice] = None
        self.model_device: Optional[str] = None
        self.audio_history: list[AudioRecord] = []
        self.latest_audio: Optional[AudioRecord] = None

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)

        self.lang_display = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
        self.display_to_lang_name = {
            lang_display_name(name): name for name in LANG_NAMES
        }

        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        reload_action = QAction("Reload Model", self)
        reload_action.triggered.connect(self.on_load_model_clicked)
        toolbar.addAction(reload_action)

        self.status_badge = QLabel("Model: Not loaded")
        self.status_badge.setObjectName("StatusBadge")
        toolbar.addWidget(self.status_badge)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        root_layout.addWidget(self.progress)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_studio_tab(), "Studio")
        tabs.addTab(self._build_batch_tab(), "Batch")
        tabs.addTab(self._build_guide_tab(), "Guide")
        root_layout.addWidget(tabs)

        self.setCentralWidget(root)

    def _build_studio_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        left_layout.addWidget(self._build_model_group())
        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_generation_group())
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(8)

        self.result_meta = QLabel("No audio generated yet.")
        self.result_meta.setObjectName("MetaText")
        self.result_meta.setWordWrap(True)
        out_layout.addWidget(self.result_meta)

        controls_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.on_play_clicked)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.save_audio_btn = QPushButton("Save As...")
        self.save_audio_btn.clicked.connect(self.on_save_audio_clicked)
        controls_row.addWidget(self.play_btn)
        controls_row.addWidget(self.stop_btn)
        controls_row.addWidget(self.save_audio_btn)
        controls_row.addStretch(1)
        out_layout.addLayout(controls_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("AppLog")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Runtime log...")
        out_layout.addWidget(self.log_view)
        right_layout.addWidget(out_group, stretch=2)

        hist_group = QGroupBox("History")
        hist_layout = QVBoxLayout(hist_group)
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.itemDoubleClicked.connect(self.on_history_double_clicked)
        hist_layout.addWidget(self.history_list)
        right_layout.addWidget(hist_group, stretch=3)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return tab

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        layout = QGridLayout(group)

        self.model_input = QLineEdit("k2-fsa/OmniVoice")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["float16", "float32"])
        self.load_asr_cb = QCheckBox("Load Whisper ASR")
        self.load_asr_cb.setChecked(False)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.on_load_model_clicked)

        layout.addWidget(QLabel("Checkpoint"), 0, 0)
        layout.addWidget(self.model_input, 0, 1, 1, 3)
        layout.addWidget(QLabel("Device"), 1, 0)
        layout.addWidget(self.device_combo, 1, 1)
        layout.addWidget(QLabel("DType"), 1, 2)
        layout.addWidget(self.dtype_combo, 1, 3)
        layout.addWidget(self.load_asr_cb, 2, 0, 1, 2)
        layout.addWidget(self.load_model_btn, 2, 3)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input")
        layout = QFormLayout(group)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setVerticalSpacing(10)

        self.text_input = QTextEdit()
        self.text_input.setObjectName("PrimaryTextInput")
        self.text_input.setPlaceholderText("Nhập nội dung cần tổng hợp giọng nói...")
        self.text_input.setMinimumHeight(190)

        self.language_combo = QComboBox()
        self.language_combo.addItems(self.lang_display)
        self.language_combo.setCurrentText("Auto")

        self.instruct_input = QLineEdit()
        self.instruct_input.setPlaceholderText(
            "vd: female, low pitch, british accent hoặc 女，青年，四川话"
        )

        self.ref_audio_input = QLineEdit()
        self.ref_audio_input.setPlaceholderText("Reference audio path (.wav/.flac/.mp3)")
        browse_audio_btn = QPushButton("Browse")
        browse_audio_btn.clicked.connect(self.on_browse_ref_audio)
        ref_audio_row = QHBoxLayout()
        ref_audio_row.addWidget(self.ref_audio_input)
        ref_audio_row.addWidget(browse_audio_btn)
        ref_audio_wrap = QWidget()
        ref_audio_wrap.setLayout(ref_audio_row)

        self.ref_text_input = QPlainTextEdit()
        self.ref_text_input.setPlaceholderText("Reference transcript (optional)")
        self.ref_text_input.setMaximumHeight(96)
        self.ref_text_input.setObjectName("RefTextInput")

        hint_mode = QLabel(
            "Chế độ tự nhận diện: có ref audio = Voice Clone, không ref audio + có instruct = Voice Design, còn lại = Auto Voice."
        )
        hint_mode.setObjectName("HintText")
        hint_mode.setWordWrap(True)

        hint_ref = QLabel(
            "Mẹo cho Voice Clone: dùng ref audio 3-10 giây, rõ tiếng, ít tạp âm. Nếu để trống Ref Text thì cần bật Load Whisper ASR."
        )
        hint_ref.setObjectName("HintText")
        hint_ref.setWordWrap(True)

        layout.addRow("Text", self.text_input)
        layout.addRow("", hint_mode)
        layout.addRow("Language", self.language_combo)
        layout.addRow("Instruct", self.instruct_input)
        layout.addRow("Ref Audio", ref_audio_wrap)
        layout.addRow("Ref Text", self.ref_text_input)
        layout.addRow("", hint_ref)
        return group

    def _build_generation_group(self) -> QGroupBox:
        group = QGroupBox("Generation")
        layout = QGridLayout(group)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Balanced", "Fast", "High Quality"])
        self.quality_combo.currentTextChanged.connect(self.apply_quality_profile)

        self.num_step = QSpinBox()
        self.num_step.setRange(4, 80)
        self.num_step.setValue(32)

        self.guidance = QDoubleSpinBox()
        self.guidance.setRange(0.0, 8.0)
        self.guidance.setSingleStep(0.1)
        self.guidance.setValue(2.0)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.2, 3.0)
        self.speed.setSingleStep(0.05)
        self.speed.setValue(1.0)

        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 120.0)
        self.duration.setSingleStep(0.5)
        self.duration.setValue(0.0)
        self.duration.setSpecialValueText("Auto")

        self.t_shift = QDoubleSpinBox()
        self.t_shift.setRange(0.01, 2.0)
        self.t_shift.setSingleStep(0.01)
        self.t_shift.setValue(0.1)

        self.position_temp = QDoubleSpinBox()
        self.position_temp.setRange(0.0, 10.0)
        self.position_temp.setSingleStep(0.1)
        self.position_temp.setValue(5.0)

        self.class_temp = QDoubleSpinBox()
        self.class_temp.setRange(0.0, 2.0)
        self.class_temp.setSingleStep(0.05)
        self.class_temp.setValue(0.0)

        self.denoise_cb = QCheckBox("Denoise")
        self.denoise_cb.setChecked(True)
        self.preprocess_prompt_cb = QCheckBox("Preprocess Prompt")
        self.preprocess_prompt_cb.setChecked(True)
        self.postprocess_output_cb = QCheckBox("Postprocess Output")
        self.postprocess_output_cb.setChecked(True)

        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.generate_btn.setToolTip("Chạy tổng hợp giọng nói với cấu hình hiện tại.")

        layout.addWidget(QLabel("Profile"), 0, 0)
        layout.addWidget(self.quality_combo, 0, 1)
        layout.addWidget(QLabel("Steps"), 0, 2)
        layout.addWidget(self.num_step, 0, 3)

        layout.addWidget(QLabel("Guidance"), 1, 0)
        layout.addWidget(self.guidance, 1, 1)
        layout.addWidget(QLabel("Speed"), 1, 2)
        layout.addWidget(self.speed, 1, 3)

        layout.addWidget(QLabel("Duration"), 2, 0)
        layout.addWidget(self.duration, 2, 1)
        layout.addWidget(QLabel("t_shift"), 2, 2)
        layout.addWidget(self.t_shift, 2, 3)

        layout.addWidget(QLabel("Position Temp"), 3, 0)
        layout.addWidget(self.position_temp, 3, 1)
        layout.addWidget(QLabel("Class Temp"), 3, 2)
        layout.addWidget(self.class_temp, 3, 3)

        layout.addWidget(self.denoise_cb, 4, 0)
        layout.addWidget(self.preprocess_prompt_cb, 4, 1)
        layout.addWidget(self.postprocess_output_cb, 4, 2)
        layout.addWidget(self.generate_btn, 4, 3)

        return group

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        group = QGroupBox("Batch Inference")
        form = QGridLayout(group)
        form.setVerticalSpacing(10)

        self.batch_model_input = QLineEdit("k2-fsa/OmniVoice")
        self.batch_test_list = QLineEdit()
        self.batch_output_dir = QLineEdit("results/desktop_batch")
        self.batch_nj_per_gpu = QSpinBox()
        self.batch_nj_per_gpu.setRange(1, 16)
        self.batch_nj_per_gpu.setValue(1)
        self.batch_batch_duration = QDoubleSpinBox()
        self.batch_batch_duration.setRange(1.0, 5000.0)
        self.batch_batch_duration.setValue(1000.0)
        self.batch_batch_size = QSpinBox()
        self.batch_batch_size.setRange(0, 512)
        self.batch_batch_size.setValue(0)

        browse_list_btn = QPushButton("Browse")
        browse_list_btn.clicked.connect(self.on_browse_batch_list)
        browse_out_btn = QPushButton("Browse")
        browse_out_btn.clicked.connect(self.on_browse_batch_output)

        self.batch_run_btn = QPushButton("Run Batch")
        self.batch_run_btn.clicked.connect(self.on_run_batch_clicked)

        form.addWidget(QLabel("Model"), 0, 0)
        form.addWidget(self.batch_model_input, 0, 1, 1, 3)
        form.addWidget(QLabel("Test JSONL"), 1, 0)
        form.addWidget(self.batch_test_list, 1, 1, 1, 2)
        form.addWidget(browse_list_btn, 1, 3)
        form.addWidget(QLabel("Output Dir"), 2, 0)
        form.addWidget(self.batch_output_dir, 2, 1, 1, 2)
        form.addWidget(browse_out_btn, 2, 3)
        form.addWidget(QLabel("Workers/GPU"), 3, 0)
        form.addWidget(self.batch_nj_per_gpu, 3, 1)
        form.addWidget(QLabel("Batch Duration"), 3, 2)
        form.addWidget(self.batch_batch_duration, 3, 3)
        form.addWidget(QLabel("Batch Size (0=auto)"), 4, 0)
        form.addWidget(self.batch_batch_size, 4, 1)
        form.addWidget(self.batch_run_btn, 4, 3)

        layout.addWidget(group)

        batch_hint = QLabel(
            "Batch phù hợp khi bạn cần tạo nhiều file từ JSONL. Tối thiểu mỗi dòng cần: id, text. Có thể bổ sung ref_audio/ref_text/instruct/language_id."
        )
        batch_hint.setObjectName("HintText")
        batch_hint.setWordWrap(True)
        layout.addWidget(batch_hint)

        self.batch_log = QPlainTextEdit()
        self.batch_log.setObjectName("AppLog")
        self.batch_log.setReadOnly(True)
        self.batch_log.setPlaceholderText("Batch logs...")
        layout.addWidget(self.batch_log, stretch=1)
        return tab

    def _build_guide_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        guide_tabs = QTabWidget()
        guide_tabs.setDocumentMode(True)
        guide_tabs.addTab(
            self._make_guide_browser(
                """
                <h2>Bắt đầu nhanh (cho mọi người)</h2>
                <p><b>Mục tiêu:</b> tạo 1 file giọng nói trong 60 giây.</p>
                <ol>
                  <li>Vào <b>Studio</b>, bấm <b>Load Model</b>.</li>
                  <li>Dán văn bản vào ô <b>Text</b>.</li>
                  <li>Chọn một cách dùng:
                    <ul>
                      <li><b>Auto Voice:</b> để trống <i>Ref Audio</i> và <i>Instruct</i>.</li>
                      <li><b>Voice Design:</b> điền <i>Instruct</i> (ví dụ: <code>female, low pitch</code>).</li>
                      <li><b>Voice Clone:</b> chọn <i>Ref Audio</i> (3-10 giây, rõ tiếng).</li>
                    </ul>
                  </li>
                  <li>Bấm <b>Generate Audio</b>, nghe lại bằng <b>Play</b>, xuất file bằng <b>Save As...</b>.</li>
                </ol>
                <h3>Mẹo chất lượng</h3>
                <ul>
                  <li>Bắt đầu với profile <b>Balanced</b>.</li>
                  <li>Muốn nhanh hơn: <b>Fast</b> (steps thấp hơn).</li>
                  <li>Muốn mượt hơn: <b>High Quality</b> (steps cao hơn).</li>
                </ul>
                """
            ),
            "Người mới",
        )
        guide_tabs.addTab(
            self._make_guide_browser(
                """
                <h2>Hướng dẫn theo nhu cầu sử dụng</h2>
                <h3>1) Giáo viên / đào tạo</h3>
                <ul>
                  <li>Dùng <b>Auto Voice</b> cho nội dung dài.</li>
                  <li>Đặt <b>duration</b> khi cần khớp thời lượng video/slides.</li>
                  <li>Dùng tiếng Việt: chọn <b>Language = Vietnamese</b> để ổn định hơn.</li>
                </ul>
                <h3>2) Content creator / social</h3>
                <ul>
                  <li>Dùng <b>Voice Design</b> để thử nhiều phong cách nhanh.</li>
                  <li>Thêm non-verbal tags như <code>[laughter]</code> để tăng biểu cảm.</li>
                  <li>Giữ đoạn text ngắn gọn, rõ nhịp để âm sắc tự nhiên hơn.</li>
                </ul>
                <h3>3) CSKH / doanh nghiệp</h3>
                <ul>
                  <li>Dùng <b>Voice Clone</b> với ref chuẩn thương hiệu.</li>
                  <li>Tạo hàng loạt bằng tab <b>Batch</b> từ JSONL.</li>
                  <li>Lưu preset nội dung bên ngoài (Notion/Sheet) rồi đẩy vào JSONL.</li>
                </ul>
                <h3>4) Kỹ thuật / MLOps</h3>
                <ul>
                  <li>Ưu tiên <b>cuda</b> nếu có GPU NVIDIA.</li>
                  <li><b>duration</b> sẽ ưu tiên hơn <b>speed</b>.</li>
                  <li>Nếu muốn giảm hậu xử lý, có thể tắt <b>postprocess_output</b>.</li>
                </ul>
                """
            ),
            "Theo vai trò",
        )
        guide_tabs.addTab(
            self._make_guide_browser(
                """
                <h2>Batch JSONL và thông số</h2>
                <p>Mỗi dòng JSONL tối thiểu:</p>
                <pre>{"id":"sample_001","text":"Xin chào"}</pre>
                <p>Trường mở rộng:</p>
                <pre>{
  "id": "sample_002",
  "text": "Nội dung cần đọc",
  "language_id": "vi",
  "instruct": "female, low pitch",
  "ref_audio": "/abs/path/ref.wav",
  "ref_text": "Nội dung bản ref",
  "duration": 8.0,
  "speed": 1.0
}</pre>
                <h3>Chọn tham số Batch</h3>
                <ul>
                  <li><b>Workers/GPU:</b> tăng khi nhiều GPU hoặc workload lớn.</li>
                  <li><b>Batch Duration:</b> tự gom mẫu theo tổng thời lượng, phù hợp dữ liệu đa dạng.</li>
                  <li><b>Batch Size &gt; 0:</b> ép số mẫu cố định mỗi batch.</li>
                </ul>
                """
            ),
            "Batch chi tiết",
        )
        guide_tabs.addTab(
            self._make_guide_browser(
                """
                <h2>Xử lý sự cố & khả năng tiếp cận</h2>
                <h3>Lỗi thường gặp</h3>
                <ul>
                  <li><b>Model load fail:</b> kiểm tra mạng hoặc checkpoint path.</li>
                  <li><b>Ref audio not found:</b> dùng đường dẫn tuyệt đối, tránh ký tự lạ.</li>
                  <li><b>Tốc độ chậm:</b> giảm <b>num_step</b> hoặc dùng profile <b>Fast</b>.</li>
                  <li><b>Thiếu module GUI:</b> cài <code>pip install "omnivoice[desktop]"</code>.</li>
                </ul>
                <h3>Khả năng tiếp cận</h3>
                <ul>
                  <li>Giao diện ưu tiên độ tương phản cao, chữ lớn, vùng bấm lớn.</li>
                  <li>Nếu vẫn khó đọc, tăng scale hệ điều hành (Display scaling) hoặc phóng to cửa sổ.</li>
                </ul>
                <h3>Nguyên tắc chọn mode</h3>
                <ul>
                  <li><b>Auto Voice:</b> nhanh, ít đầu vào.</li>
                  <li><b>Voice Design:</b> chủ động phong cách.</li>
                  <li><b>Voice Clone:</b> bám chất giọng tham chiếu tốt nhất.</li>
                </ul>
                """
            ),
            "Troubleshooting",
        )
        layout.addWidget(guide_tabs)
        return tab

    def _make_guide_browser(self, html: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        return browser

    def _apply_theme(self):
        QApplication.setStyle("Fusion")
        font = QFont("Inter", 12)
        QApplication.instance().setFont(font)
        self.setStyleSheet(
            """
            QWidget {
                color: #0F172A;
            }
            QMainWindow {
                background-color: #EEF2F7;
            }
            QToolBar {
                background: #0F172A;
                border: none;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton {
                color: #E2E8F0;
                font-weight: 700;
                background: transparent;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QToolButton:hover {
                background: #1E293B;
            }
            QGroupBox {
                border: 1px solid #CBD5E1;
                border-radius: 14px;
                margin-top: 12px;
                padding: 10px;
                background: #FFFFFF;
                font-weight: 700;
                color: #0F172A;
                font-size: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background: #0B7FA1;
                border: 1px solid #0B7FA1;
                color: white;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 700;
                min-height: 22px;
            }
            QPushButton:hover {
                background: #0A6D8A;
                border-color: #0A6D8A;
            }
            QPushButton:disabled {
                background: #94A3B8;
                border-color: #94A3B8;
                color: #F8FAFC;
            }
            QLabel {
                color: #0F172A;
                font-size: 14px;
            }
            QLabel#HintText {
                color: #334155;
                font-size: 13px;
                background: #F8FAFC;
                border: 1px dashed #CBD5E1;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel#MetaText {
                color: #0F172A;
                font-size: 14px;
                background: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                padding: 10px;
            }
            QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #A5B4C7;
                border-radius: 10px;
                background: #FFFFFF;
                padding: 8px;
                color: #0F172A;
                selection-background-color: #D1FAE5;
                selection-color: #052E16;
                font-size: 14px;
            }
            QTextEdit#PrimaryTextInput {
                font-size: 16px;
                line-height: 1.4;
            }
            QPlainTextEdit#RefTextInput {
                font-size: 14px;
            }
            QPlainTextEdit#AppLog {
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 13px;
                background: #0B1220;
                color: #E5E7EB;
                border: 1px solid #1F2937;
            }
            QTabWidget::pane {
                border: 1px solid #CBD5E1;
                border-radius: 12px;
                background: #F8FAFC;
            }
            QTabBar::tab {
                background: #E2E8F0;
                color: #0B1324;
                border: 1px solid #B8C4D3;
                border-bottom: none;
                padding: 10px 16px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 3px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                font-weight: 700;
            }
            QListWidget {
                background: #FFFFFF;
                border: 1px solid #A5B4C7;
                border-radius: 10px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background: #DBEAFE;
                color: #0C4A6E;
            }
            QCheckBox {
                color: #0F172A;
                font-size: 14px;
                font-weight: 600;
                spacing: 8px;
                padding: 3px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 2px solid #334155;
                background: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background: #0B7FA1;
                border-color: #0B7FA1;
            }
            QProgressBar {
                border: 1px solid #94A3B8;
                border-radius: 6px;
                background: #E2E8F0;
            }
            QProgressBar::chunk {
                background: #0B7FA1;
                border-radius: 5px;
            }
            QLabel#StatusBadge {
                margin-left: 10px;
                padding: 6px 12px;
                border: 1px solid #67E8F9;
                border-radius: 10px;
                background: #164E63;
                color: #ECFEFF;
                font-weight: 700;
                font-size: 14px;
            }
            """
        )

    def _set_busy(self, busy: bool):
        self.progress.setMaximum(0 if busy else 1)
        self.progress.setValue(0 if busy else 1)
        self.load_model_btn.setDisabled(busy)
        self.generate_btn.setDisabled(busy)
        self.batch_run_btn.setDisabled(busy)

    def _append_log(self, text: str):
        self.log_view.appendPlainText(text)

    def apply_quality_profile(self, *_):
        profile = self.quality_combo.currentText()
        if profile == "Fast":
            self.num_step.setValue(16)
            self.guidance.setValue(1.8)
        elif profile == "High Quality":
            self.num_step.setValue(48)
            self.guidance.setValue(2.2)
        else:
            self.num_step.setValue(32)
            self.guidance.setValue(2.0)

    def on_browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.m4a);;All Files (*)",
        )
        if path:
            self.ref_audio_input.setText(path)

    def on_browse_batch_list(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSONL", "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if path:
            self.batch_test_list.setText(path)

    def on_browse_batch_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.batch_output_dir.setText(path)

    def on_load_model_clicked(self):
        checkpoint = self.model_input.text().strip()
        if not checkpoint:
            QMessageBox.warning(self, "Invalid Input", "Checkpoint path is required.")
            return

        device_choice = self.device_combo.currentText()
        target_device = best_device() if device_choice == "auto" else device_choice
        dtype = torch.float16 if self.dtype_combo.currentText() == "float16" else torch.float32
        load_asr = self.load_asr_cb.isChecked()

        worker = FunctionWorker(
            self._load_model_impl, checkpoint, target_device, dtype, load_asr
        )
        worker.signals.started.connect(partial(self._set_busy, True))
        worker.signals.finished.connect(self._on_load_model_success)
        worker.signals.failed.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _load_model_impl(
        self, checkpoint: str, device: str, dtype: torch.dtype, load_asr: bool
    ):
        model = OmniVoice.from_pretrained(
            checkpoint, device_map=device, dtype=dtype, load_asr=load_asr
        )
        return model, device, checkpoint

    def _on_load_model_success(self, payload: object):
        self._set_busy(False)
        model, device, checkpoint = payload  # type: ignore[misc]
        self.model = model
        self.model_device = device
        self.status_badge.setText(f"Model: Ready ({device})")
        self.batch_model_input.setText(checkpoint)
        self._append_log(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded: {checkpoint} on {device}")

    def on_generate_clicked(self):
        if self.model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load model first.")
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Invalid Input", "Text is required.")
            return

        display_lang = self.language_combo.currentText()
        language = None
        if display_lang != "Auto":
            language = self.display_to_lang_name.get(display_lang, display_lang)

        instruct = self.instruct_input.text().strip() or None
        ref_audio = self.ref_audio_input.text().strip() or None
        ref_text = self.ref_text_input.toPlainText().strip() or None
        if ref_audio and not os.path.exists(ref_audio):
            QMessageBox.warning(self, "Invalid Input", f"Reference audio not found:\n{ref_audio}")
            return

        mode = "Auto Voice"
        if ref_audio:
            mode = "Voice Clone"
        elif instruct:
            mode = "Voice Design"

        kwargs = dict(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            duration=None if self.duration.value() <= 0 else float(self.duration.value()),
            num_step=int(self.num_step.value()),
            guidance_scale=float(self.guidance.value()),
            speed=float(self.speed.value()),
            t_shift=float(self.t_shift.value()),
            denoise=self.denoise_cb.isChecked(),
            preprocess_prompt=self.preprocess_prompt_cb.isChecked(),
            postprocess_output=self.postprocess_output_cb.isChecked(),
            position_temperature=float(self.position_temp.value()),
            class_temperature=float(self.class_temp.value()),
        )
        worker = FunctionWorker(self._generate_impl, kwargs, mode)
        worker.signals.started.connect(partial(self._set_busy, True))
        worker.signals.finished.connect(self._on_generate_success)
        worker.signals.failed.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _generate_impl(self, kwargs: dict[str, Any], mode: str):
        assert self.model is not None
        audios = self.model.generate(**kwargs)
        audio = audios[0]
        sample_rate = int(self.model.sampling_rate)
        fd, wav_path = tempfile.mkstemp(prefix="omnivoice_desktop_", suffix=".wav")
        os.close(fd)
        sf.write(wav_path, audio, sample_rate)
        duration_sec = len(audio) / sample_rate
        text = kwargs["text"]
        preview = text if len(text) <= 80 else f"{text[:80]}..."
        record = AudioRecord(
            created_at=datetime.now(),
            mode=mode,
            text_preview=preview,
            wav_path=wav_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
        )
        return record

    def _on_generate_success(self, payload: object):
        self._set_busy(False)
        record = payload  # type: ignore[assignment]
        assert isinstance(record, AudioRecord)
        self.latest_audio = record
        self.audio_history.insert(0, record)

        self.result_meta.setText(
            f"Mode: {record.mode}\n"
            f"Duration: {record.duration_sec:.2f}s @ {record.sample_rate}Hz\n"
            f"Text: {record.text_preview}"
        )

        item = QListWidgetItem(
            f"{record.created_at.strftime('%H:%M:%S')} | {record.mode} | {record.duration_sec:.2f}s"
        )
        item.setData(Qt.UserRole, record.wav_path)
        self.history_list.insertItem(0, item)
        self._append_log(
            f"[{record.created_at.strftime('%H:%M:%S')}] Generated {record.mode} ({record.duration_sec:.2f}s)"
        )
        self.on_play_clicked()

    def _on_worker_error(self, trace: str):
        self._set_busy(False)
        self._append_log(trace)
        QMessageBox.critical(self, "Runtime Error", trace.splitlines()[-1] if trace else "Unknown error")

    def on_play_clicked(self):
        if not self.latest_audio:
            return
        self.media_player.setSource(QUrl.fromLocalFile(self.latest_audio.wav_path))
        self.media_player.play()

    def on_stop_clicked(self):
        self.media_player.stop()

    def on_save_audio_clicked(self):
        if not self.latest_audio:
            QMessageBox.information(self, "No Audio", "No generated audio available.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio",
            f"omnivoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            "WAV Files (*.wav)",
        )
        if not save_path:
            return
        with open(self.latest_audio.wav_path, "rb") as src, open(save_path, "wb") as dst:
            dst.write(src.read())
        self._append_log(f"Saved audio: {save_path}")

    def on_history_double_clicked(self, item: QListWidgetItem):
        wav_path = item.data(Qt.UserRole)
        if not wav_path or not os.path.exists(wav_path):
            return
        self.media_player.setSource(QUrl.fromLocalFile(wav_path))
        self.media_player.play()

    def on_run_batch_clicked(self):
        test_list = self.batch_test_list.text().strip()
        if not test_list:
            QMessageBox.warning(self, "Invalid Input", "Batch JSONL path is required.")
            return
        if not os.path.exists(test_list):
            QMessageBox.warning(self, "Invalid Input", f"JSONL not found:\n{test_list}")
            return
        output_dir = self.batch_output_dir.text().strip() or "results/desktop_batch"
        os.makedirs(output_dir, exist_ok=True)
        model = self.batch_model_input.text().strip() or "k2-fsa/OmniVoice"

        args = [
            sys.executable,
            "-m",
            "omnivoice.cli.infer_batch",
            "--model",
            model,
            "--test_list",
            test_list,
            "--res_dir",
            output_dir,
            "--nj_per_gpu",
            str(self.batch_nj_per_gpu.value()),
            "--batch_duration",
            str(self.batch_batch_duration.value()),
        ]
        if self.batch_batch_size.value() > 0:
            args.extend(["--batch_size", str(self.batch_batch_size.value())])

        worker = FunctionWorker(self._run_batch_impl, args)
        worker.signals.started.connect(partial(self._set_busy, True))
        worker.signals.finished.connect(self._on_batch_done)
        worker.signals.failed.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _run_batch_impl(self, args: list[str]):
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        return {"returncode": proc.returncode, "output": proc.stdout, "cmd": args}

    def _on_batch_done(self, payload: object):
        self._set_busy(False)
        result = payload  # type: ignore[assignment]
        assert isinstance(result, dict)
        cmd = " ".join(result["cmd"])
        self.batch_log.appendPlainText(f"$ {cmd}\n")
        self.batch_log.appendPlainText(result["output"])
        if result["returncode"] == 0:
            self.batch_log.appendPlainText("\nBatch inference completed successfully.")
        else:
            self.batch_log.appendPlainText(
                f"\nBatch inference failed with return code {result['returncode']}."
            )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OmniVoice Studio")
    app.setWindowIcon(QIcon())
    window = OmniVoiceDesktop()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

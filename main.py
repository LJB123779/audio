import sys
import os
import time
import tempfile
import shutil
from typing import List, Tuple
import json
import re
import urllib.request
import urllib.error


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg
import numpy as np
from pydub import AudioSegment

SUPPORTED_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a'}

# App versioning and GitHub repo info (replace with your repo)
APP_VERSION = '1.0'
GITHUB_REPO = 'LJB123779/audio'  # e.g., 'octocat/Hello-World'

# Silence marker helpers for targeted insertion
SILENCE_MARKER_PREFIX = '__SILENCE__:('

def is_silence_marker(entry: str) -> bool:
	try:
		return isinstance(entry, str) and entry.startswith(SILENCE_MARKER_PREFIX) and entry.endswith(')')
	except Exception:
		return False

def make_silence_marker(duration_ms: int) -> str:
	if duration_ms < 0:
		duration_ms = 0
	return f"{SILENCE_MARKER_PREFIX}{int(duration_ms)})"

def parse_silence_marker_ms(entry: str) -> int:
	if not is_silence_marker(entry):
		return 0
	try:
		return int(entry[len(SILENCE_MARKER_PREFIX):-1])
	except Exception:
		return 0

# UI display helpers
def _format_seconds_from_ms(ms: int) -> str:
	try:
		sec = round(ms / 1000.0, 2)
		if abs(sec - int(sec)) < 1e-9:
			return f"{int(sec)}s"
		text = f"{sec:.2f}".rstrip('0').rstrip('.')
		return f"{text}s"
	except Exception:
		return '0s'

def to_display_text(entry: str) -> str:
	if is_silence_marker(entry):
		ms = parse_silence_marker_ms(entry)
		return f"静音{_format_seconds_from_ms(ms)}"
	return os.path.basename(entry)


class MergeWorkerThread(QtCore.QThread):
	# 信号定义
	progress_updated = QtCore.pyqtSignal(int)  # 进度更新 (0-100)
	merge_completed = QtCore.pyqtSignal(object, np.ndarray, np.ndarray)  # (merged_audio, x_data, y_data)
	merge_error = QtCore.pyqtSignal(str)  # 错误信息
	
	def __init__(self, file_list: List[str]):
		super().__init__()
		self.file_list = file_list
		self._cancel_requested = False
	
	def cancel(self):
		self._cancel_requested = True
	
	def run(self):
		try:
			merged = None
			total_files = len(self.file_list)
			
			for idx, f in enumerate(self.file_list):
				if self._cancel_requested:
					return
				
				# 更新进度
				progress = int((idx / total_files) * 80)  # 合并占80%进度
				self.progress_updated.emit(progress)
				
				# If this entry is a silence marker, insert targeted silence and continue
				if is_silence_marker(f):
					marker_ms = parse_silence_marker_ms(f)
					if marker_ms > 0:
						if merged is None:
							merged = AudioSegment.silent(duration=marker_ms)
						else:
							merged += AudioSegment.silent(duration=marker_ms)
					continue

				# Regular audio file
				seg = AudioSegment.from_file(f)
				if merged is None:
					merged = seg
				else:
					merged += seg
			
			if self._cancel_requested:
				return
			
			# 处理音频数据用于波形显示 (占剩余20%进度)
			self.progress_updated.emit(85)
			
			sample_width = merged.sample_width
			channels = merged.channels
			frame_rate = merged.frame_rate
			raw = np.array(merged.get_array_of_samples())
			
			if self._cancel_requested:
				return
			
			self.progress_updated.emit(90)
			
			if channels > 1:
				raw = raw.reshape((-1, channels))
				raw = raw.mean(axis=1)
			
			raw = raw.astype(np.float32)
			max_val = np.max(np.abs(raw)) or 1.0
			raw /= max_val
			
			if self._cancel_requested:
				return
			
			self.progress_updated.emit(95)
			
			x = np.linspace(0, len(raw) / frame_rate, num=len(raw))
			
			self.progress_updated.emit(100)
			
			# 发送完成信号
			self.merge_completed.emit(merged, x, raw)
			
		except Exception as e:
			self.merge_error.emit(str(e))


class UpdateCheckThread(QtCore.QThread):
	# 成功时返回 {tag_name, html_url, name, body}
	success = QtCore.pyqtSignal(dict)
	# 失败时返回错误信息
	error = QtCore.pyqtSignal(str)

	def __init__(self, repo: str, parent=None):
		super().__init__(parent)
		self.repo = repo

	def run(self):
		try:
			if not self.repo or '/' not in self.repo:
				raise ValueError('无效的 GitHub 仓库标识')
			url = f'https://api.github.com/repos/{self.repo}/releases/latest'
			req = urllib.request.Request(url, headers={'User-Agent': 'audio2-updater'})
			with urllib.request.urlopen(req, timeout=10) as resp:
				data = json.loads(resp.read().decode('utf-8', errors='ignore'))
			info = {
				'tag_name': data.get('tag_name') or '',
				'html_url': data.get('html_url') or '',
				'name': data.get('name') or '',
				'body': data.get('body') or '',
			}
			self.success.emit(info)
		except Exception as ex:
			self.error.emit(str(ex))


def is_audio_file(path: str) -> bool:
	_, ext = os.path.splitext(path)
	ext = ext.lower()
	return ext in SUPPORTED_EXTS


class DraggableListWidget(QtWidgets.QListWidget):
	filesDropped = QtCore.pyqtSignal(list)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setAcceptDrops(True)
		self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

	def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
		if event.mimeData().hasUrls():
			event.acceptProposedAction()
		else:
			event.ignore()

	def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
		if event.mimeData().hasUrls():
			event.acceptProposedAction()
		else:
			event.ignore()

	def dropEvent(self, event: QtGui.QDropEvent):
		urls = event.mimeData().urls()
		dropped_paths = []
		for url in urls:
			path = url.toLocalFile()
			if os.path.isdir(path):
				for root, _, files in os.walk(path):
					for name in files:
						full = os.path.join(root, name)
						if is_audio_file(full):
							dropped_paths.append(full)
			elif os.path.isfile(path) and is_audio_file(path):
				dropped_paths.append(path)
		if dropped_paths:
			self.filesDropped.emit(dropped_paths)
			event.acceptProposedAction()
		else:
			event.ignore()


class AudioMergerApp(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('音频合并与预览工具')
		self.resize(1100, 700)

		self.file_list: List[str] = []
		# removed global between-all silence control; use targeted markers only
		self.merged: AudioSegment = None
		self.preview_path: str = ''
		self.position_ms = 0
		self._timer = QtCore.QTimer(self)
		self._timer.setInterval(50)
		self._timer.timeout.connect(self._on_timer_tick)
		self._was_playing_before_seek = False
		
		# 工作线程和进度对话框
		self.merge_worker = None
		self.progress_dialog = None

		# settings
		self.settings = QtCore.QSettings('audio2', 'audio_merger')
		self.ffmpeg_path = self.settings.value('ffmpeg_path', type=str) or ''
		self.app_version = APP_VERSION
		self.github_repo = GITHUB_REPO
		self._update_thread = None
		self._update_manual_trigger = False

		# Qt 媒体播放器用于预览
		self.player = QMediaPlayer(self)
		self.player.setVolume(100)
		self.player.positionChanged.connect(self._on_player_position)
		self.player.durationChanged.connect(self._on_player_duration)
		self.player.stateChanged.connect(self._on_player_state)

		self._build_ui()
		# 启动时自动检查更新（每日一次）
		self._auto_check_for_updates_if_due()

	def _build_ui(self):
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)

		# Menu: Settings -> FFmpeg path
		menubar = self.menuBar()
		menu_settings = menubar.addMenu('设置')
		act_ffmpeg = QtWidgets.QAction('设置FFmpeg路径...', self)
		act_ffmpeg.triggered.connect(self._on_set_ffmpeg)
		menu_settings.addAction(act_ffmpeg)

		# Menu: Help -> Check for updates
		menu_help = menubar.addMenu('帮助')
		act_check_update = QtWidgets.QAction('检查更新...', self)
		act_check_update.triggered.connect(lambda: self._check_for_updates(True))
		menu_help.addAction(act_check_update)

		# Left: files and controls
		left_layout = QtWidgets.QVBoxLayout()

		self.list_widget = DraggableListWidget()
		self.list_widget.filesDropped.connect(self._on_files_dropped)
		left_layout.addWidget(QtWidgets.QLabel('文件列表（可拖拽文件/文件夹）：'))
		left_layout.addWidget(self.list_widget, 1)

		btns_layout = QtWidgets.QHBoxLayout()
		self.btn_add = QtWidgets.QPushButton('添加文件')
		self.btn_remove = QtWidgets.QPushButton('删除所选')
		self.btn_clear = QtWidgets.QPushButton('清空')
		btns_layout.addWidget(self.btn_add)
		btns_layout.addWidget(self.btn_remove)
		btns_layout.addWidget(self.btn_clear)
		left_layout.addLayout(btns_layout)

		# Targeted silence insertion controls
		insert_silence_layout = QtWidgets.QHBoxLayout()
		insert_silence_layout.addWidget(QtWidgets.QLabel('插入静音（秒）：'))
		self.insert_silence_spin = QtWidgets.QDoubleSpinBox()
		self.insert_silence_spin.setRange(0.0, 300.0)
		self.insert_silence_spin.setSingleStep(0.5)
		self.insert_silence_spin.setDecimals(2)
		self.insert_silence_spin.setValue(1.00)
		insert_silence_layout.addWidget(self.insert_silence_spin)
		self.btn_insert_silence = QtWidgets.QPushButton('在所选后插入静音')
		insert_silence_layout.addWidget(self.btn_insert_silence)
		left_layout.addLayout(insert_silence_layout)

		# removed global between-all silence UI

		self.btn_merge = QtWidgets.QPushButton('合并')
		self.btn_export = QtWidgets.QPushButton('导出MP3')
		self.btn_merge.clicked.connect(self._on_merge)
		self.btn_export.clicked.connect(self._on_export)
		left_layout.addWidget(self.btn_merge)
		left_layout.addWidget(self.btn_export)

		# Right: preview and waveform
		right_layout = QtWidgets.QVBoxLayout()
		right_layout.addWidget(QtWidgets.QLabel('预览（合并后）：'))

		self.plot_widget = pg.PlotWidget()
		self.plot_widget.setBackground('w')
		self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
		self.wave_plot = self.plot_widget.plot(pen=pg.mkPen('#007acc', width=1))
		right_layout.addWidget(self.plot_widget, 1)

		controls_layout = QtWidgets.QHBoxLayout()
		self.btn_play = QtWidgets.QPushButton('播放')
		self.btn_pause = QtWidgets.QPushButton('暂停')
		self.btn_stop = QtWidgets.QPushButton('停止')
		controls_layout.addWidget(self.btn_play)
		controls_layout.addWidget(self.btn_pause)
		controls_layout.addWidget(self.btn_stop)
		right_layout.addLayout(controls_layout)

		seek_layout = QtWidgets.QHBoxLayout()
		self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.seek_slider.setRange(0, 0)
		self.seek_slider.sliderPressed.connect(self._on_seek_start)
		self.seek_slider.sliderReleased.connect(self._on_seek_end)
		self.seek_slider.sliderMoved.connect(self._on_seek_moved)
		self.lbl_time = QtWidgets.QLabel('00:00 / 00:00')
		seek_layout.addWidget(self.seek_slider, 1)
		seek_layout.addWidget(self.lbl_time)
		right_layout.addLayout(seek_layout)

		# Volume meter (RMS-like bar)
		self.volume_bar = QtWidgets.QProgressBar()
		self.volume_bar.setRange(0, 100)
		right_layout.addWidget(self.volume_bar)

		# Layout split
		main_layout = QtWidgets.QHBoxLayout()
		main_layout.addLayout(left_layout, 3)
		main_layout.addLayout(right_layout, 5)
		central.setLayout(main_layout)

		# Signals
		self.btn_add.clicked.connect(self._on_add_files)
		self.btn_remove.clicked.connect(self._on_remove_selected)
		self.btn_clear.clicked.connect(self._on_clear)
		self.btn_play.clicked.connect(self._on_play)
		self.btn_pause.clicked.connect(self._on_pause)
		self.btn_stop.clicked.connect(self._on_stop)
		self.btn_insert_silence.clicked.connect(self._on_insert_silence_after_selected)

	def _auto_check_for_updates_if_due(self):
		try:
			last_ts = int(self.settings.value('last_update_check_ts', 0))
		except Exception:
			last_ts = 0
		now_ts = int(time.time())
		if now_ts - last_ts >= 24 * 3600:
			self._check_for_updates(False)

	def _check_for_updates(self, manual: bool = False):
		# 防止并发检查
		if self._update_thread and self._update_thread.isRunning():
			return
		self._update_manual_trigger = manual
		self._update_thread = UpdateCheckThread(self.github_repo, self)
		self._update_thread.success.connect(self._on_update_success)
		self._update_thread.error.connect(self._on_update_error)
		self._update_thread.start()

	@staticmethod
	def _normalize_version_text(text: str) -> str:
		if not text:
			return '0.0.0'
		# 去掉前缀 'v' 等，并提取数字与点
		text = text.strip()
		if text.lower().startswith('v'):
			text = text[1:]
		parts = re.findall(r'\d+', text)
		if not parts:
			return '0.0.0'
		return '.'.join(parts)

	@staticmethod
	def _version_tuple(ver: str) -> Tuple[int, int, int, int]:
		parts = [int(p) for p in ver.split('.') if p.isdigit()]
		# pad to length 4 for safe comparison
		while len(parts) < 4:
			parts.append(0)
		return tuple(parts[:4])  # type: ignore

	def _is_remote_newer(self, local: str, remote: str) -> bool:
		lv = self._version_tuple(self._normalize_version_text(local))
		rv = self._version_tuple(self._normalize_version_text(remote))
		return rv > lv

	def _on_update_success(self, info: dict):
		# 更新检查时间
		self.settings.setValue('last_update_check_ts', int(time.time()))
		remote_tag = info.get('tag_name') or ''
		remote_url = info.get('html_url') or ''
		if self._is_remote_newer(self.app_version, remote_tag):
			ret = QtWidgets.QMessageBox.question(
				self,
				'发现新版本',
				f'检测到新版本：{remote_tag}\n当前版本：{self.app_version}\n是否前往 GitHub 下载？',
				QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
				QtWidgets.QMessageBox.Yes)
			if ret == QtWidgets.QMessageBox.Yes and remote_url:
				QtGui.QDesktopServices.openUrl(QtCore.QUrl(remote_url))
		else:
			if self._update_manual_trigger:
				QtWidgets.QMessageBox.information(self, '检查更新', '当前已是最新版本。')
		# 清理线程引用
		if self._update_thread:
			self._update_thread.deleteLater()
			self._update_thread = None

	def _on_update_error(self, msg: str):
		# 仅在手动检查时反馈错误，自动检查静默失败
		if self._update_manual_trigger:
			QtWidgets.QMessageBox.warning(self, '检查更新失败', f'无法获取最新版本信息：\n{msg}')
		# 清理线程引用
		if self._update_thread:
			self._update_thread.deleteLater()
			self._update_thread = None

	def _on_files_dropped(self, paths: List[str]):
		for p in paths:
			self.file_list.append(p)
			self.list_widget.addItem(to_display_text(p))

	def _on_add_files(self):
		files, _ = QtWidgets.QFileDialog.getOpenFileNames(
			self, '选择音频文件', os.path.expanduser('~'),
			'音频文件 (*.mp3 *.wav *.flac *.ogg *.aac *.m4a)')
		for f in files:
			self.file_list.append(f)
			self.list_widget.addItem(to_display_text(f))

	def _on_remove_selected(self):
		for item in self.list_widget.selectedItems():
			row = self.list_widget.row(item)
			self.list_widget.takeItem(row)
			self.file_list.pop(row)

	def _on_insert_silence_after_selected(self):
		# Insert a targeted silence marker after each selected row (from bottom to top to keep indices stable)
		selected = self.list_widget.selectedIndexes()
		if not selected:
			QtWidgets.QMessageBox.information(self, '提示', '请先选择需要在其后插入静音的条目')
			return
		duration_sec = float(self.insert_silence_spin.value())
		if duration_sec <= 0:
			QtWidgets.QMessageBox.information(self, '提示', '静音时长应大于 0 毫秒')
			return
		duration_ms = int(round(duration_sec * 1000))
		# Sort descending by row
		rows = sorted([idx.row() for idx in selected], reverse=True)
		for row in rows:
			marker = make_silence_marker(duration_ms)
			self.file_list.insert(row + 1, marker)
			self.list_widget.insertItem(row + 1, to_display_text(marker))
		# Keep selection on original items
		self.list_widget.clearSelection()
		for row in rows:
			self.list_widget.item(row).setSelected(True)

	def _on_clear(self):
		self.list_widget.clear()
		self.file_list.clear()
	
	def _create_progress_dialog(self) -> QtWidgets.QProgressDialog:
		"""创建进度对话框"""
		dialog = QtWidgets.QProgressDialog('正在合并音频文件...', '取消', 0, 100, self)
		dialog.setWindowTitle('合并进度')
		dialog.setWindowModality(QtCore.Qt.WindowModal)
		dialog.setMinimumDuration(0)
		dialog.canceled.connect(self._on_merge_canceled)
		return dialog
	
	def _on_merge_canceled(self):
		"""用户取消合并操作"""
		if self.merge_worker:
			self.merge_worker.cancel()
			self.merge_worker.wait(3000)  # 等待最多3秒
			if self.merge_worker.isRunning():
				self.merge_worker.terminate()
		self._cleanup_merge_operation()
	
	def _cleanup_merge_operation(self):
		"""清理合并操作相关资源"""
		if self.progress_dialog:
			self.progress_dialog.close()
			self.progress_dialog = None
		if self.merge_worker:
			self.merge_worker.deleteLater()
			self.merge_worker = None
		# 重新启用UI控件
		self._set_ui_enabled(True)
	
	def _set_ui_enabled(self, enabled: bool):
		"""启用/禁用UI控件"""
		self.btn_merge.setEnabled(enabled)
		self.btn_add.setEnabled(enabled)
		self.btn_remove.setEnabled(enabled)
		self.btn_clear.setEnabled(enabled)
		self.list_widget.setEnabled(enabled)
		# removed global between-all silence control
		self.insert_silence_spin.setEnabled(enabled)
		self.btn_insert_silence.setEnabled(enabled)
	
	def _on_merge_progress(self, value: int):
		"""更新合并进度"""
		if self.progress_dialog:
			self.progress_dialog.setValue(value)
	
	def _on_merge_completed(self, merged_audio: AudioSegment, x_data: np.ndarray, y_data: np.ndarray):
		"""合并完成处理"""
		try:
			# 更新波形显示
			self.wave_plot.setData(x_data, y_data)
			self.plot_widget.setLabel('bottom', '时间', units='s')
			self.plot_widget.setYRange(-1.05, 1.05)
			
			self.merged = merged_audio
			self.position_ms = 0
			self.seek_slider.setRange(0, int(len(self.merged)))
			self._update_time_label()
			
			# 生成预览 WAV 并载入播放器
			tmp_dir = tempfile.gettempdir()
			self.preview_path = os.path.join(tmp_dir, 'audio2_preview.wav')
			try:
				self.merged.export(self.preview_path, format='wav')
				self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(self.preview_path)))
			except Exception as ex:
				QtWidgets.QMessageBox.warning(self, '提示', f'生成预览文件失败：{ex}')
			
			QtWidgets.QMessageBox.information(self, '成功', '合并完成，可预览或导出。')
		finally:
			self._cleanup_merge_operation()
	
	def _on_merge_error(self, error_msg: str):
		"""合并错误处理"""
		QtWidgets.QMessageBox.critical(self, '错误', f'合并失败：{error_msg}')
		self._cleanup_merge_operation()

	def _on_merge(self):
		if not self.file_list:
			QtWidgets.QMessageBox.warning(self, '提示', '请先添加音频文件')
			return
		if not self._ensure_ffmpeg():
			return
		
		# 如果已经有合并操作在进行，则忽略
		if self.merge_worker and self.merge_worker.isRunning():
			return
		
		# 禁用UI控件
		self._set_ui_enabled(False)
		
		# 创建并显示进度对话框
		self.progress_dialog = self._create_progress_dialog()
		self.progress_dialog.show()
		
		# 创建并启动工作线程
		self.merge_worker = MergeWorkerThread(self.file_list.copy())
		self.merge_worker.progress_updated.connect(self._on_merge_progress)
		self.merge_worker.merge_completed.connect(self._on_merge_completed)
		self.merge_worker.merge_error.connect(self._on_merge_error)
		self.merge_worker.start()

	def _on_export(self):
		if self.merged is None:
			QtWidgets.QMessageBox.warning(self, '提示', '请先合并音频')
			return
		if not self._ensure_ffmpeg():
			return
		path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, '导出MP3', os.path.expanduser('~'), 'MP3 文件 (*.mp3)')
		if not path:
			return
		if not path.lower().endswith('.mp3'):
			path += '.mp3'
		try:
			self.merged.export(path, format='mp3')
			QtWidgets.QMessageBox.information(self, '成功', f'已导出：\n{path}')
		except Exception as e:
			QtWidgets.QMessageBox.critical(self, '错误', f'导出失败：{e}\n请确认已安装 FFmpeg 并在 PATH 中。')

	def _stop_playback(self):
		self.player.stop()
		if self._timer.isActive():
			self._timer.stop()

	def _on_play(self):
		if self.merged is None:
			QtWidgets.QMessageBox.warning(self, '提示', '请先合并音频')
			return
		if self.preview_path:
			if self.player.mediaStatus() == QMediaPlayer.NoMedia:
				self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(self.preview_path)))
			if self.position_ms > 0:
				self.player.setPosition(self.position_ms)
			self.player.play()

	def _on_pause(self):
		self.player.pause()

	def _on_stop(self):
		self._stop_playback()
		self.position_ms = 0
		self._update_seek_visuals()

	def _on_seek_start(self):
		self._was_playing_before_seek = (self.player.state() == QMediaPlayer.PlayingState)
		self._stop_playback()

	def _on_seek_end(self):
		self.position_ms = int(self.seek_slider.value())
		self._update_time_label()
		if self._was_playing_before_seek and self.merged is not None:
			self.player.setPosition(self.position_ms)
			self.player.play()
		self._was_playing_before_seek = False

	def _on_seek_moved(self, value: int):
		self.position_ms = int(value)
		self._update_time_label()

	def _on_timer_tick(self):
		if self.merged is None:
			return
		self.position_ms = int(self.player.position())
		self._update_seek_visuals()
		try:
			chunk = self.merged[max(0, self.position_ms - 50):self.position_ms]
			self._update_volume_meter(chunk)
		except Exception:
			pass

	def _on_set_ffmpeg(self):
		start_dir = os.path.dirname(self.ffmpeg_path) if self.ffmpeg_path else os.path.expanduser('~')
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择 ffmpeg.exe', start_dir, '可执行文件 (*.exe)')
		if not path:
			return
		if not os.path.isfile(path) or not path.lower().endswith('ffmpeg.exe'):
			QtWidgets.QMessageBox.warning(self, '提示', '请选择 ffmpeg.exe')
			return
		self.ffmpeg_path = path
		self.settings.setValue('ffmpeg_path', self.ffmpeg_path)
		self._apply_ffmpeg_path()
		QtWidgets.QMessageBox.information(self, '成功', 'FFmpeg 路径已设置。')

	def _apply_ffmpeg_path(self):
		# Configure pydub to use custom ffmpeg
		try:
			if self.ffmpeg_path and os.path.isfile(self.ffmpeg_path):
				AudioSegment.converter = self.ffmpeg_path
				AudioSegment.ffmpeg = self.ffmpeg_path
				ffprobe_guess = os.path.join(os.path.dirname(self.ffmpeg_path), 'ffprobe.exe')
				if os.path.isfile(ffprobe_guess):
					AudioSegment.ffprobe = ffprobe_guess
		except Exception:
			pass

	def _ensure_ffmpeg(self) -> bool:
		# if path set and exists
		if self.ffmpeg_path and os.path.isfile(self.ffmpeg_path):
			self._apply_ffmpeg_path()
			return True
		# try PATH
		found = shutil.which('ffmpeg')
		if found:
			self.ffmpeg_path = found
			self._apply_ffmpeg_path()
			return True
		# try winget installed FFmpeg
		winget_ffmpeg_path = r"C:\Users\35264\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
		if os.path.isfile(winget_ffmpeg_path):
			self.ffmpeg_path = winget_ffmpeg_path
			self._apply_ffmpeg_path()
			return True
		# ask user to set
		ret = QtWidgets.QMessageBox.question(
			self, '需要 FFmpeg', '未检测到 FFmpeg，是否现在设置 ffmpeg.exe 路径？',
			QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
		if ret == QtWidgets.QMessageBox.Yes:
			self._on_set_ffmpeg()
			return bool(self.ffmpeg_path and os.path.isfile(self.ffmpeg_path))
		else:
			return False

	def _on_player_position(self, pos: int):
		self.position_ms = int(pos)
		self._update_seek_visuals()

	def _on_player_duration(self, dur: int):
		self.seek_slider.setRange(0, int(dur))
		self._update_time_label()

	def _on_player_state(self, state: int):
		if state == QMediaPlayer.PlayingState:
			if not self._timer.isActive():
				self._timer.start()
		else:
			if self._timer.isActive():
				self._timer.stop()

	def _update_seek_visuals(self):
		self.seek_slider.blockSignals(True)
		self.seek_slider.setValue(int(self.position_ms))
		self.seek_slider.blockSignals(False)
		self._update_time_label()

	def _update_time_label(self):
		cur = self._format_ms(self.position_ms)
		total = self._format_ms(int(len(self.merged)) if self.merged is not None else 0)
		self.lbl_time.setText(f'{cur} / {total}')

	@staticmethod
	def _format_ms(ms: int) -> str:
		sec = ms // 1000
		m = sec // 60
		s = sec % 60
		return f'{m:02d}:{s:02d}'

	def _update_volume_meter(self, chunk: AudioSegment):
		arr = np.array(chunk.get_array_of_samples()).astype(np.float32)
		if chunk.channels > 1:
			arr = arr.reshape((-1, chunk.channels)).mean(axis=1)
		# RMS normalized to 0..100 range
		if arr.size == 0:
			value = 0
		else:
			rms = np.sqrt(np.mean(np.square(arr)))
			peak = np.max(np.abs(arr)) or 1.0
			value = int(min(100, (rms / peak) * 100))
		self.volume_bar.setValue(value)


def main():
	app = QtWidgets.QApplication(sys.argv)
	win = AudioMergerApp()
	win.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()

"""
Main GUI – Miruro-style monotone dark media player + YOLO pipeline.
Controls overlay the video. Fixed 1.5× base scale (no dynamic rescaling).
True fullscreen with auto-hiding transport controls.
"""
from __future__ import annotations
import os, queue, tempfile, time
from typing import Dict, Optional, TYPE_CHECKING
import cv2, numpy as np, qtawesome as qta
from PyQt5 import QtCore, QtGui, QtWidgets
from config import TARGET_CLASS_IDS, CLASS_NAMES, VIDEO_EXTENSIONS, MODEL_EXTENSIONS
from utils.model_registry import discover_models, _detect_format, _FORMAT_LABEL, cleanup_onnx_artifacts
from pipeline.state import PipelineState
from utils.runtime_check import detect as _detect_runtimes
from utils.camera_devices import discover_camera_devices, open_camera_capture
from gui.widgets import VideoDropLabel, SeekSlider
from pipeline import supabase_logger as _sblog

if TYPE_CHECKING:
    from pipeline.frame_grabber import FrameGrabberThread
    from pipeline.inference_engine import InferenceThread
    from pipeline.tracker_logic import TrackerLogicThread
    from pipeline.convert_worker import ConvertWorker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Palette ──────────────────────────────────────────────────────────────────
_BG = "#0d0d0d"; _BG_SB = "#0f0f0f"; _BD = "#1e1e1e"
_T = "#b0b0b0"; _TD = "#555555"; _TH = "#e0e0e0"; _W = "#ffffff"
_HOV = "rgba(255,255,255,0.07)"; _PRS = "rgba(255,255,255,0.14)"
_R = 6          # base border-radius in logical px (multiplied by _S)

def _fa(name, color=_T, sz=18):
    return qta.icon(f"fa5s.{name}", color=color)


_S = 1.5  # fixed UI scale - restore missing global
_DROP_PROMPT = "Drop a video here or use the sidebar to load one"
_FS_ICON_SIZE = int(16 * _S)

# ── Supabase project (anon key — RLS-protected) ──────────────────────────────
_SUPABASE_URL = "https://aenoakqaattuokctmpou.supabase.co"
_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFlbm9ha3FhYXR0dW9rY3RtcG91Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3OTI4MzEsImV4cCI6MjA5MjM2ODgzMX0."
    "w0ROs4P8LmDoKVAZnW9J59RSIktT1eLa5Uek2_VctD4"
)

_VIOLATION_LABELS = {
    "no_helmet": "NO HELMET",
    "improper_footwear": "BAD FOOTWEAR",
    "overload": "OVERLOAD",
}

# Sentinel key used by the model-picker dictionaries.  The system uses a
# single unified detector for every class id, so model_combo / browse /
# convert widgets are stored once under this key instead of once per class.
_UNIFIED_KEY = -1
_STAT_ITEMS = (
    ("motorcycles", "Motorcycles"),
    ("riders", "Riders"),
    ("helmet_unknown", "Helmet ?"),
    ("footwear_unknown", "Footwear ?"),
    ("improper_footwear", "Bad Footwear"),
    ("overloaded_motos", "Overload"),
    ("invalid_detections", "Occluded"),
)
_CLASS_INFER_RULES = (
    (4, ("improper", "no footwear", "barefoot")),
    (3, ("footwear", "shoe", "boot")),
    (2, ("helmet", "hardhat")),
    (0, ("motorcycle", "motorbike", "moto")),
    (1, ("rider", "driver", "person")),
)
_CLASS_INFER_TRANSLATE = str.maketrans("-_.", "   ")


class _AlignedComboBox(QtWidgets.QComboBox):
    """QComboBox that shows a speed-menu-style QMenu popup instead of the
    native Qt dropdown, giving full style control."""

    def showPopup(self) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.NoDropShadowWindowHint
        )
        menu.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        _mr     = int(_R * _S)
        _hpad   = int(10 * _S)
        _vpad   = int(4  * _S)   # matches combo's vertical padding
        _item_h = int(20 * _S)   # matches combo's min-height → same 42 px total
        menu.setStyleSheet(
            f"QMenu{{background:#161616;color:{_T};"
            f"border:1px solid {_BD};border-radius:{_mr}px;"
            f"padding:0;}}"
            f"QMenu::item{{padding:{_vpad}px {_hpad}px;"
            f"font-size:{int(10*_S)}px;"
            f"font-family:'Consolas','Courier New',monospace;"
            f"min-height:{_item_h}px;}}"
            f"QMenu::item:selected{{background:#2a2a2a;color:{_TH};}}"
            f"QMenu::item:checked{{background:#2a2a2a;color:{_TH};}}"
            f"QMenu::indicator{{width:0;height:0;}}"
        )
        fm = menu.fontMetrics()
        max_text_w = max(
            (fm.horizontalAdvance(self.itemText(i))
             for i in range(self.count())),
            default=0,
        )
        popup_w = max(self.width(), max_text_w + _hpad * 2 + int(8 * _S))
        menu.setFixedWidth(popup_w)
        for i in range(self.count()):
            act = menu.addAction(self.itemText(i))
            act.setData(i)
            act.setCheckable(True)
            act.setChecked(i == self.currentIndex())
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        chosen = menu.exec_(pos)
        if chosen is not None:
            self.setCurrentIndex(chosen.data())


class _CenteredComboBox(_AlignedComboBox):
    """A combo box whose closed-state display draws its icon + current text
    centered (matching the look of the *Load Video* button)."""

    def paintEvent(self, event) -> None:  # noqa: D401  (Qt method)
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        # Wipe icon + text so the native control just paints the chrome
        # (border, background, drop-down arrow); we render the contents
        # ourselves below.
        opt.currentText = ""
        opt.currentIcon = QtGui.QIcon()
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)

        # ── Centered icon + text ────────────────────────────────────────
        icon = self.itemIcon(self.currentIndex()) if self.count() > 0 else QtGui.QIcon()
        text = self.currentText()
        spacing = 6
        icon_size = 0
        pix: Optional[QtGui.QPixmap] = None
        if not icon.isNull():
            icon_size = max(self.iconSize().width(), 14)
            pix = icon.pixmap(icon_size, icon_size)
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        total_w = text_w + (icon_size + spacing if pix is not None else 0)
        rect = self.rect()
        # Reserve a bit of room on the right for the drop-down arrow
        avail_w = rect.width() - 18
        x = max(rect.left() + 4, rect.left() + (avail_w - total_w) // 2)
        y = rect.center().y()
        if pix is not None:
            target = QtCore.QRect(x, y - icon_size // 2, icon_size, icon_size)
            painter.drawPixmap(target, pix)
            x += icon_size + spacing
        painter.setPen(self.palette().color(QtGui.QPalette.ButtonText))
        text_rect = QtCore.QRect(x, rect.top(), text_w, rect.height())
        painter.drawText(
            text_rect,
            int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft),
            text,
        )


class _CameraScanThread(QtCore.QThread):
    """Background camera discovery to keep UI responsive."""

    completed = QtCore.pyqtSignal(list)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        max_index: int = 8,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._max_index = max(1, int(max_index))

    def run(self) -> None:
        try:
            devices = discover_camera_devices(max_index=self._max_index)
            payload = [d.to_dict() for d in devices]
            self.completed.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))

def _qss(s: float, chk: str) -> str:
    """Full stylesheet – every pixel value multiplied by *s*."""
    def px(v): return f"{max(1, round(v * s))}px"
    _r = px(_R)  # consistent corner radius
    _r_sm = px(max(1, _R - 2))  # smaller radius for tiny elements
    # dot handle: seek slider custom‐painted, sidebar drawn by QSS
    _dot = px(8)          # sidebar handle diameter
    _dot_r = px(4)        # sidebar handle radius = diameter / 2
    _dot_m = px(3)        # negative margin so handle extends beyond groove
    return (
        f"*{{outline:0;}}"
        f"QMainWindow,QWidget{{background:{_BG};color:{_T};"
        f"font-family:'Segoe UI',sans-serif;font-size:{px(12)};}}"
        f"#topBar{{background:{_BG};border-bottom:1px solid {_BD};}}"
        f"#topBar QLabel{{background:transparent;}}"
        f"#topBarTitle{{color:{_TH};font-size:{px(13)};font-weight:600;"
        f"background:transparent;}}"
        f"#sidebar{{background:{_BG_SB};border-right:1px solid {_BD};}}"
        f"#sidebar QLabel{{color:{_T};}}"
        f"#sidebarTitle{{font-size:{px(13)};font-weight:600;color:{_TH};"
        f"padding:{px(6)} 0 {px(10)} 0;margin:0;letter-spacing:0.5px;}}"
        f"#sectionLabel{{font-size:{px(10)};font-weight:600;color:{_TD};"
        f"text-transform:uppercase;"
        f"padding:{px(14)} 0 {px(4)} 0;margin:0;"
        f"qproperty-indent:0;}}"
        f"#modelCard{{background:transparent;border:none;"
        f"border-radius:0;padding:{px(2)} 0;margin:0;}}"
        f"#modelFileIcon{{background:transparent;border:none;min-width:0;"
        f"padding:0;margin:0;}}"
        f"#modelFileIcon:hover{{background:rgba(255,255,255,0.06);"
        f"border-radius:{_r_sm};}}"
        f"#modelFileIcon:focus{{outline:none;border:none;background:transparent;}}"
        f"#modelFile{{color:{_TD};font-size:{px(11)};"
        f"padding:{px(2)} {px(6)};background:transparent;}}"
        f"#modelFileLoaded{{color:{_TH};font-size:{px(11)};"
        f"padding:{px(2)} {px(6)};background:transparent;}}"
        f"#videoFileRow{{background:transparent;border:none;"
        f"border-radius:0;}}"
        f"#videoFileRow *{{background:transparent;}}"
        f"#videoFileRow:hover{{background:rgba(255,255,255,0.04);}}"
        # buttons
        f"QPushButton{{background:{_BG};color:{_T};border:1px solid {_BD};"
        f"border-radius:{_r};padding:{px(5)} {px(12)};"
        f"font-size:{px(11)};}}"
        f"QPushButton:hover{{background:#1a1a1a;}}"
        f"QPushButton:pressed{{background:#222;}}"
        f"QPushButton:disabled{{background:{_BG};color:#333;border-color:#151515;}}"
        f"QPushButton:focus{{outline:none;border-color:{_BD};background:{_BG};}}"
        f"#iconBtn{{background:transparent;border:none;"
        f"border-radius:{_r};padding:0;outline:none;}}"
        f"#iconBtn:hover{{background:{_HOV};}}"
        f"#iconBtn:pressed{{background:{_PRS};}}"
        f"#iconBtn:focus{{outline:none;border:none;background:transparent;}}"
        f"#closeVideoBtn{{background:transparent;border:none;"
        f"padding:0;margin:0;min-width:0;}}"
        f"#closeVideoBtn:hover{{background:rgba(255,255,255,0.08);"
        f"border-radius:{_r_sm};}}"
        f"#closeVideoBtn:focus{{outline:none;border:none;background:transparent;}}"
        # seek slider – groove only; circle drawn by SeekSlider.paintEvent
        f"QSlider#seekSlider::groove:horizontal{{height:{px(4)};"
        f"background:#4d4d4d;border:none;border-radius:{px(2)};}}"
        f"QSlider#seekSlider::sub-page:horizontal{{background:#f5f5f5;"
        f"border-radius:{px(2)};}}"
        f"QSlider#seekSlider::add-page:horizontal{{background:#4d4d4d;"
        f"border-radius:{px(2)};}}"
        f"QSlider#seekSlider::handle:horizontal{{"
        f"background:transparent;border:none;"
        f"width:{px(14)};height:{px(14)};"
        f"margin:-{px(5)} 0;border-radius:{px(7)};}}"
        # sidebar sliders – circular handle matching seekbar dot size
        f"QSlider::groove:horizontal{{height:{px(3)};background:#2a2a2a;"
        f"border-radius:{px(1)};}}"
        f"QSlider::handle:horizontal{{background:{_TH};border:none;"
        f"width:{_dot};height:{_dot};margin:-{_dot_m} 0;"
        f"border-radius:{_dot_r};}}"
        f"QSlider::handle:horizontal:hover{{background:{_W};}}"
        f"QSlider::sub-page:horizontal{{background:{_T};"
        f"border-radius:{px(1)};}}"
        # checkboxes
        f"QCheckBox{{spacing:{px(6)};color:{_T};font-size:{px(11)};}}"
        f"QCheckBox::indicator{{width:{px(13)};height:{px(13)};"
        f"border:1px solid #444;border-radius:{_r_sm};background:#1a1a1a;}}"
        f"QCheckBox::indicator:checked{{background:#1a1a1a;"
        f"border:1px solid #555;border-radius:{_r_sm};"
        f"image:url({chk});}}"
        # video display
        f"#videoDisplay{{background:#000;border:none;color:{_TD};"
        f"font-size:{px(13)};}}"
        f"#overlayCtrl,#overlaySeek,#ctrlBar{{background:transparent;}}"
        f"#overlayCtrl QWidget{{background:transparent;}}"
        f"#timeLabel{{color:rgba(255,255,255,0.6);font-size:{px(11)};"
        f"font-family:'Consolas','Courier New',monospace;"
        f"padding:0;margin:0;background:transparent;}}"
        f"#vidName{{color:{_T};font-size:{px(11)};padding:0;"
        f"background:transparent;}}"
        f"#sliderLabel{{color:{_TD};font-size:{px(11)};margin:0;"
        f"padding:0;qproperty-indent:0;}}"
        # comboboxes – styled to match the playback-speed QMenu popup
        f"QComboBox{{background:#161616;color:{_T};"
        f"border:1px solid {_BD};border-radius:{_r};"
        f"padding:{px(4)} {px(8)};font-size:{px(10)};"
        f"font-family:'Consolas','Courier New',monospace;"
        f"min-height:{px(20)};}}"
        f"QComboBox:hover{{border-color:#333;}}"
        f"QComboBox:on{{border-color:#333;}}"
        f"QComboBox#loadVideoCombo{{font-family:inherit;font-size:{px(11)};background:{_BG};border:1px solid {_BD};border-radius:{px(max(1, round(_R * s)))};}}"
        f"QComboBox#loadVideoCombo:hover{{background:#1a1a1a;}}"
        f"QComboBox::drop-down{{border:none;width:{px(18)};}}"
        f"QComboBox::down-arrow{{image:none;}}"
        f"QComboBox QAbstractItemView{{background:#161616;color:{_T};"
        f"border:1px solid {_BD};border-radius:{_r};"
        f"selection-background-color:#2a2a2a;"
        f"selection-color:{_TH};outline:0;"
        f"font-family:'Consolas','Courier New',monospace;"
        f"font-size:{px(11)};padding:{px(4)} 0;}}"
        f"QComboBox QAbstractItemView::viewport{{background:#161616;"
        f"border-radius:{_r};}}"
        f"QComboBox QAbstractItemView::item{{"
        f"padding:{px(4)} {px(10)};min-height:{px(28)};color:{_T};}}"
        f"QComboBox QAbstractItemView::item:hover{{"
        f"background:#2a2a2a;color:{_TH};}}"
        f"QComboBox QAbstractItemView::item:selected{{"
        f"background:#2a2a2a;color:{_TH};}}"
        f"QComboBox QAbstractItemView::item:selected:active{{"
        f"background:#2a2a2a;color:{_TH};}}"
        f"QComboBox QAbstractItemView::item:selected:!active{{"
        f"background:#2a2a2a;color:{_TH};}}"
        f"QFrame#comboPopup{{background:#161616;border:1px solid {_BD};"
        f"border-radius:{_r};}}"
        f"QComboBox#modelCombo{{padding:{px(4)} {px(8)};}}"
        f"QComboBox#modelCombo QAbstractItemView::item{{"
        f"padding:{px(4)} {px(10)};min-height:{px(28)};color:{_T};}}"
        # status
        f"#statusLabel{{color:#444;font-size:{px(10)};"
        f"padding:{px(1)} {px(8)};}}"
        f"QScrollArea{{border:none;background:transparent;}}"
        f"QScrollBar:vertical{{background:transparent;width:{px(6)};"
        f"margin:0;}}"
        f"QScrollBar::handle:vertical{{background:#2a2a2a;"
        f"border-radius:{px(3)};min-height:{px(30)};}}"
        f"QScrollBar::handle:vertical:hover{{background:#3a3a3a;}}"
        f"QScrollBar::add-line:vertical,"
        f"QScrollBar::sub-line:vertical{{height:0;}}"
        f"QScrollBar:horizontal{{height:0;background:transparent;}}"
        # tooltips
        f"QToolTip{{background:#1a1a1a;color:{_T};border:1px solid #333;"
        f"border-radius:{_r};padding:{px(4)} {px(8)};font-size:{px(11)};}}"
        # colour dot in sidebar (QPushButton overrides)
        f"#colorDot{{background:transparent;border:none;"
        f"border-radius:{px(5)};"
        f"min-width:{px(10)};max-width:{px(10)};"
        f"min-height:{px(10)};max-height:{px(10)};"
        f"padding:0;margin:0;}}"
        f"#colorDot:hover{{border:none;}}"
        f"#colorDot:pressed{{border:none;}}"
        f"#colorDot:focus{{border:none;outline:none;}}"
    )


class MainWindow(QtWidgets.QMainWindow):
    """Miruro-style monotone dark media player + YOLO pipeline controller."""

    def __init__(self) -> None:
        super().__init__()
        self.resize(1630, 760)
        self.setAcceptDrops(True)
        self._chk = self._make_checkmark_icon()
        self._state = PipelineState()
        self._state.set_overlay_enabled(True)
        self._grabber: Optional["FrameGrabberThread"] = None
        self._inferencer: Optional["InferenceThread"] = None
        self._tracker: Optional["TrackerLogicThread"] = None
        self._timer = QtCore.QTimer(self, interval=16)
        self._timer.timeout.connect(self._poll_display)
        self._video_fps = self._current_fps = 0.0
        self._total_frames = 0
        self._user_seeking = self._is_playing = False
        self._last_pixmap = QtGui.QPixmap()
        self._last_rgb_frame: Optional[np.ndarray] = None
        self._preview_cap: Optional[cv2.VideoCapture] = None
        # fullscreen state
        self._is_fs = False
        self._fs_hide = QtCore.QTimer(self)
        self._cam_scan: Optional[_CameraScanThread] = None
        # model conversion worker
        self._converter: Optional["ConvertWorker"] = None
        self._cvt_t0: float = 0.0          # monotonic start time of conversion
        self._cvt_label: str = ""           # last "Converting X → Y" message
        # model registry state
        self._model_groups: Dict = {}
        self._prev_combo_idx: Dict[int, int] = {}
        # heartbeat timer: update status bar every 20 s during long TRT builds
        self._cvt_timer = QtCore.QTimer(self)
        self._cvt_timer.setInterval(20_000)
        self._cvt_timer.timeout.connect(self._on_cvt_heartbeat)
        self._fs_hide.setSingleShot(True)
        self._fs_hide.setInterval(3000)
        self._fs_hide.timeout.connect(self._fs_hide_overlay)
        # YouTube-style HUD auto-hide (when the eye toggle is "closed").
        # Always created, but only started while _hud_locked is False.
        self._hud_idle_timer = QtCore.QTimer(self)
        self._hud_idle_timer.setSingleShot(True)
        self._hud_idle_timer.setInterval(2000)
        self._hud_idle_timer.timeout.connect(self._hud_auto_hide)
        self._build_ui()
        self._setup_shortcuts()
        self.setStyleSheet(_qss(_S, self._chk))
        self._auto_load_models()
        QtCore.QTimer.singleShot(120, self._refresh_camera_devices)
        # Initialise Supabase compliance logger (background thread).
        try:
            _sblog.configure(
                _SUPABASE_URL, _SUPABASE_ANON_KEY,
                status_cb=lambda lvl, msg: QtCore.QTimer.singleShot(
                    0, lambda lvl=lvl, msg=msg: self._log(msg, level=lvl)),
            )
            self._log("Supabase logger ready — DB: compliance_logs")
        except Exception as e:
            self._log_error(f"Supabase init failed: {e}")
        self._timer.start()

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_checkmark_icon() -> str:
        path = os.path.join(tempfile.gettempdir(), "sgc_chk.png")
        pm = QtGui.QPixmap(13, 13); pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(_W), 1.8)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        p.setPen(pen); p.drawLine(3, 7, 5, 10); p.drawLine(5, 10, 10, 3)
        p.end(); pm.save(path, "PNG")
        return path.replace("\\", "/")

    def _ibtn(self, name: str, tip: str, sz: int = 16) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton()
        isz = int(sz * _S)
        bsz = int(34 * _S)  # fixed button box size
        b.setIcon(_fa(name, _W, isz))
        b.setIconSize(QtCore.QSize(isz, isz))
        b.setFixedSize(bsz, bsz)
        b.setObjectName("iconBtn"); b.setToolTip(tip); b.setFlat(True)
        b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        b.setFocusPolicy(QtCore.Qt.NoFocus)
        return b

    @staticmethod
    def _section(t: str) -> QtWidgets.QLabel:
        l = QtWidgets.QLabel(t); l.setObjectName("sectionLabel"); return l

    @staticmethod
    def _to_qcolor_bgr(color_bgr: tuple[int, int, int]) -> QtGui.QColor:
        b, g, r = color_bgr
        return QtGui.QColor(r, g, b)

    @staticmethod
    def _to_bgr_tuple(color: QtGui.QColor) -> tuple[int, int, int]:
        return (color.blue(), color.green(), color.red())

    def _has_cuda(self) -> bool:
        return bool(_detect_runtimes().has_cuda)

    def _has_active_source(self) -> bool:
        return self._state.has_source()

    def _video_scale_mode(self) -> QtCore.Qt.AspectRatioMode:
        # Always preserve the source aspect ratio.  Using
        # ``KeepAspectRatioByExpanding`` cropped portions of the annotated
        # frame off-screen, which hid bounding boxes drawn near the edges.
        return QtCore.Qt.KeepAspectRatio

    def _fit_window_to_video_aspect(self, src_w: int, src_h: int) -> None:
        """Resize window width so main video viewport matches source AR."""
        if src_w <= 0 or src_h <= 0 or self._is_fs:
            return
        src_ar = src_w / float(src_h)
        top_h = self._topBar.height() if hasattr(self, "_topBar") else 0
        status_h = self.statusLabel.height() if hasattr(self, "statusLabel") else 0
        main_h = max(180, self.height() - top_h - status_h)
        target_main_w = int(round(main_h * src_ar))
        sidebar_w = self.sidebar.width() if hasattr(self, "sidebar") else int(260 * _S)
        handle_w = self._splitter.handleWidth() if hasattr(self, "_splitter") else 1
        target_total_w = target_main_w + sidebar_w + handle_w
        target_total_w = max(self.minimumWidth(), target_total_w)
        self.resize(target_total_w, self.height())

    # ── ui ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._splitter.setHandleWidth(1)
        self._splitter.setStyleSheet(
            f"QSplitter::handle{{background:{_BD};}}")
        root.addWidget(self._splitter)

        self.sidebar = self._build_sidebar()
        self.railbar = self._build_railbar()
        self._sidebarStack = QtWidgets.QStackedWidget()
        self._sidebarStack.addWidget(self.sidebar)   # index 0 - expanded
        self._sidebarStack.addWidget(self.railbar)   # index 1 - collapsed
        self._sidebar_collapsed = False
        self._splitter.addWidget(self._sidebarStack)

        main = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(main)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)
        self._splitter.addWidget(main)

        # Right-side console / logs panel (hidden by default)
        self._consolePanel = self._build_console_panel()
        self._splitter.addWidget(self._consolePanel)
        self._consolePanel.hide()

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(2, False)
        # Console panel default width is 75% of the legacy 360px value.
        self._console_default_w = int(270 * _S)
        self._console_last_w = 0  # populated when user resizes / closes
        self._splitter.setSizes(
            [int(260 * _S), int(805 * _S), self._console_default_w])

        def _on_splitter_moved(*_a):
            sizes = self._splitter.sizes()
            if len(sizes) >= 3 and self._consolePanel.isVisible() and sizes[2] > 0:
                self._console_last_w = sizes[2]
        self._splitter.splitterMoved.connect(_on_splitter_moved)

        # top bar
        self._topBar = tb = QtWidgets.QWidget()
        tb.setObjectName("topBar"); tb.setFixedHeight(int(38 * _S))
        tr = QtWidgets.QHBoxLayout(tb)
        tr.setContentsMargins(int(12*_S), 0, int(10*_S), 0)
        tr.setSpacing(int(8 * _S))

        # Hamburger – toggles sidebar between full and collapsed (rail).
        self.btnHamburger = self._ibtn("bars", "Toggle sidebar")
        tr.addWidget(self.btnHamburger, 0, QtCore.Qt.AlignVCenter)

        title = QtWidgets.QLabel("Safety Gear Compliance")
        title.setObjectName("topBarTitle")
        tr.addWidget(title); tr.addStretch()

        # Eye toggle – when "open" the playback HUD is pinned visible.
        # When "closed" the HUD auto-hides like a YouTube player.
        self._hud_locked = True
        self.btnHudLock = self._ibtn("eye", "HUD pinned (click to auto-hide)")
        tr.addWidget(self.btnHudLock, 0, QtCore.Qt.AlignVCenter)

        # Logs / console panel toggle
        self.btnLogs = self._ibtn("terminal", "Show console (logs)")
        tr.addWidget(self.btnLogs, 0, QtCore.Qt.AlignVCenter)

        vbox.addWidget(tb)

        # video container
        self._vc = QtWidgets.QWidget()
        self._vc.setObjectName("videoContainer")
        self._vc.setStyleSheet("#videoContainer{background:#000;}")
        self._vc.setMouseTracking(True)
        cl = QtWidgets.QVBoxLayout(self._vc)
        cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        self.videoDisplay = VideoDropLabel(
            _DROP_PROMPT)
        self.videoDisplay.setObjectName("videoDisplay")
        self.videoDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.videoDisplay.setMinimumSize(320, 180)
        self.videoDisplay.setScaledContents(False)
        self.videoDisplay.setMouseTracking(True)
        cl.addWidget(self.videoDisplay, stretch=1)
        vbox.addWidget(self._vc, stretch=1)

        # overlay (gradient backdrop)
        self._overlay = QtWidgets.QWidget(self._vc)
        self._overlay.setObjectName("overlayCtrl")
        self._overlay.setAttribute(
            QtCore.Qt.WA_TransparentForMouseEvents, False)
        self._overlay.setMouseTracking(True)
        self._overlay.setStyleSheet(
            "#overlayCtrl{background:qlineargradient("
            "x1:0,y1:0,x2:0,y2:1,"
            "stop:0 rgba(0,0,0,0),stop:0.3 rgba(0,0,0,80),"
            "stop:1 rgba(0,0,0,200));}")
        olay = QtWidgets.QVBoxLayout(self._overlay)
        olay.setContentsMargins(0, 0, 0, 0); olay.setSpacing(0)

        # seek bar – centered inside the overlay hitbox
        skW = QtWidgets.QWidget(); skW.setObjectName("overlaySeek")
        # height = 2 × hover-circle-radius (+4 px breathing room) × scale
        _sk_h = int((8 * 2 + 4) * _S)  # = 30 px at 1.5×
        skW.setFixedHeight(_sk_h)
        _sl_h = int(14 * _S)
        _vpad = max(0, (_sk_h - _sl_h) // 2)
        sl = QtWidgets.QHBoxLayout(skW)
        sl.setContentsMargins(int(12*_S), _vpad, int(12*_S), _vpad)
        sl.setSpacing(0)
        self.seekSlider = SeekSlider(scale=_S)
        self.seekSlider.setObjectName("seekSlider")
        self.seekSlider.setFixedHeight(_sl_h)
        self.seekSlider.setRange(0, 0); self.seekSlider.setEnabled(False)
        self.seekSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        sl.addWidget(self.seekSlider)
        olay.addWidget(skW)

        # seek preview tooltip
        self._seekPrev = prev = QtWidgets.QWidget(self._vc)
        prev.setObjectName("seekPreview")
        prev.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        prev.setStyleSheet(
            "#seekPreview{background:rgba(0,0,0,0.85);"
            "border:1px solid #333;border-radius:4px;}"
            "#seekPreview QLabel{background:transparent;color:#fff;}")
        pvl = QtWidgets.QVBoxLayout(prev)
        pvl.setContentsMargins(6, 4, 6, 4); pvl.setSpacing(0)
        self._prevImg = None  # thumbnail disabled – time-only tooltip
        self._prevTime = QtWidgets.QLabel("0:00")
        self._prevTime.setAlignment(QtCore.Qt.AlignCenter)
        self._prevTime.setStyleSheet(
            f"font-size:{int(11*_S)}px;"
            "font-family:'Consolas','Courier New',monospace;")
        pvl.addWidget(self._prevTime)
        prev.adjustSize(); prev.hide()

        # control bar
        self._ctrlBar = cb = QtWidgets.QWidget()
        cb.setObjectName("ctrlBar"); cb.setFixedHeight(int(50 * _S))
        cr = QtWidgets.QHBoxLayout(cb)
        cr.setContentsMargins(int(10*_S), int(6*_S), int(10*_S), int(6*_S))
        cr.setSpacing(int(4 * _S))
        AV = QtCore.Qt.AlignVCenter

        psz = int(16 * _S)
        self._ico_play = _fa("play", _W, psz)
        self._ico_pause = _fa("pause", _W, psz)
        self.btnPlay = self._ibtn("play", "Play / Pause")
        self.btnStepFwd = self._ibtn("step-forward", "Next frame")
        self._btnVol = self._ibtn("volume-up", "Volume")

        # speed slider (replaces menu button)
        self._speeds = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        self._spd_idx = 2  # default 1.0×
        _spd_icon = QtWidgets.QLabel()
        _spd_isz = int(13 * _S)
        _spd_icon.setPixmap(_fa("tachometer-alt", _TD, _spd_isz).pixmap(_spd_isz, _spd_isz))
        _spd_icon.setFixedSize(_spd_isz, _spd_isz)
        _spd_icon.setStyleSheet("background:transparent;")
        self.spdSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spdSlider.setObjectName("speedSlider")
        self.spdSlider.setRange(0, len(self._speeds) - 1)
        self.spdSlider.setValue(self._spd_idx)
        self.spdSlider.setFixedWidth(int(70 * _S))
        self.spdSlider.setFixedHeight(int(14 * _S))
        self.spdSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.spdSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.spdLabel = QtWidgets.QLabel("1×")
        self.spdLabel.setObjectName("timeLabel")
        self.spdLabel.setFixedWidth(int(28 * _S))

        for w in (self.btnPlay, self.btnStepFwd, self._btnVol):
            cr.addWidget(w, 0, AV)
        cr.addSpacing(int(4 * _S))
        cr.addWidget(_spd_icon, 0, AV)
        cr.addSpacing(int(4 * _S))
        cr.addWidget(self.spdSlider, 0, AV)
        cr.addWidget(self.spdLabel, 0, AV)

        sp = QtWidgets.QWidget(); sp.setFixedWidth(int(8 * _S))
        sp.setStyleSheet("background:transparent;")
        cr.addWidget(sp)

        self.lblTime = QtWidgets.QLabel("0:00")
        self.lblTime.setObjectName("timeLabel")
        sep = QtWidgets.QLabel("/"); sep.setObjectName("timeLabel")
        sep.setFixedWidth(int(8 * _S))
        sep.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTotal = QtWidgets.QLabel("0:00")
        self.lblTotal.setObjectName("timeLabel")
        for w in (self.lblTime, sep, self.lblTotal):
            cr.addWidget(w, 0, AV)
        cr.addStretch()

        self.btnRew = self._ibtn("undo-alt", "Rewind 10 s")
        self.btnFwd = self._ibtn("redo-alt", "Forward 10 s")
        self.btnShot = self._ibtn("camera", "Screenshot")
        self.btnFS = self._ibtn("expand", "Fullscreen")
        for b in (self.btnRew, self.btnFwd, self.btnShot, self.btnFS):
            cr.addWidget(b, 0, AV)
        olay.addWidget(cb)

        # status
        self.statusLabel = QtWidgets.QLabel("Ready")
        self.statusLabel.setObjectName("statusLabel")
        self.statusLabel.setFixedHeight(int(16 * _S))
        vbox.addWidget(self.statusLabel)

        self._connect_signals()
        self._vc.installEventFilter(self)
        self.videoDisplay.installEventFilter(self)
        self.seekSlider.setMouseTracking(True)
        self.seekSlider.installEventFilter(self)
        self._overlay.installEventFilter(self)
        self._splitter.splitterMoved.connect(
            lambda: QtCore.QTimer.singleShot(0, self._repos))
        QtCore.QTimer.singleShot(0, self._repos)

    # ── sidebar ───────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sb = QtWidgets.QWidget(); sb.setObjectName("sidebar")
        sb.setMinimumWidth(int(180 * _S))
        sb.setMaximumWidth(int(300 * _S))
        outer = QtWidgets.QVBoxLayout(sb)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(
            int(14*_S), int(14*_S), int(14*_S), int(14*_S))
        lay.setSpacing(int(8 * _S))

        hdr = QtWidgets.QLabel("Controls")
        hdr.setObjectName("sidebarTitle")
        lay.addWidget(hdr)

        # VIDEO / DEVICE
        vid_sec_lay = QtWidgets.QHBoxLayout()
        vid_sec_lay.setContentsMargins(0, int(14*_S), 0, int(4*_S))
        vid_sec_lay.setSpacing(int(4 * _S))
        vid_sec_lbl = self._section("VIDEO / DEVICE")
        vid_sec_lbl.setContentsMargins(0, 0, 0, 0)
        # Override the CSS padding so the layout margin drives the spacing,
        # letting the button align vertically with the label text.
        vid_sec_lbl.setStyleSheet(
            f"color:{_TD};font-size:{int(10*_S)}px;font-weight:600;"
            "text-transform:uppercase;padding:0;margin:0;"
            "qproperty-indent:0;"
        )
        vid_sec_lay.addWidget(vid_sec_lbl, 1, QtCore.Qt.AlignVCenter)

        cam_icon_sz = int(11 * _S)
        self.btnRefreshCams = QtWidgets.QPushButton()
        self.btnRefreshCams.setObjectName("iconBtn")
        self.btnRefreshCams.setFlat(True)
        self.btnRefreshCams.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnRefreshCams.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnRefreshCams.setToolTip("Refresh camera list")
        self.btnRefreshCams.setIcon(_fa("sync-alt", _TD, cam_icon_sz))
        self.btnRefreshCams.setFixedSize(int(22 * _S), int(22 * _S))
        vid_sec_lay.addWidget(self.btnRefreshCams, 0, QtCore.Qt.AlignVCenter)
        
        lay.addLayout(vid_sec_lay)

        self.btnLoadVideo = QtWidgets.QPushButton()
        isz = int(13 * _S)
        self.btnLoadVideo.setIcon(_fa("folder-open", _T, isz))
        self.btnLoadVideo.setIconSize(QtCore.QSize(isz, isz))
        self.btnLoadVideo.setText("  Load Video")
        self.btnLoadVideo.setFixedHeight(int(28 * _S))
        self.btnLoadVideo.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnLoadVideo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self.btnLoadVideo)

        self._vfWidget = QtWidgets.QWidget()
        self._vfWidget.setObjectName("videoFileRow")
        vfr = QtWidgets.QHBoxLayout(self._vfWidget)
        vfr.setContentsMargins(
            int(8*_S), int(4*_S), int(4*_S), int(4*_S))
        vfr.setSpacing(int(4 * _S))
        fsz = int(12 * _S)
        vfi = QtWidgets.QLabel()
        vfi.setPixmap(_fa("film", _TD, fsz).pixmap(fsz, fsz))
        vfi.setFixedSize(int(14 * _S), int(14 * _S))
        vfr.addWidget(vfi)
        self.lblVidName = QtWidgets.QLabel("")
        self.lblVidName.setObjectName("vidName")
        self.lblVidName.setWordWrap(False)
        self.lblVidName.setMinimumWidth(0)
        self.lblVidName.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        vfr.addWidget(self.lblVidName, stretch=1)
        _sq = int(24 * _S)  # square button size
        csz = int(11 * _S)
        self.btnCloseVid = QtWidgets.QPushButton()
        self.btnCloseVid.setIcon(_fa("times", _TD, csz))
        self.btnCloseVid.setIconSize(QtCore.QSize(csz, csz))
        self.btnCloseVid.setObjectName("closeVideoBtn")
        self.btnCloseVid.setToolTip("Close video")
        self.btnCloseVid.setFlat(True)
        self.btnCloseVid.setFixedSize(_sq, _sq)
        self.btnCloseVid.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnCloseVid.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        vfr.addWidget(self.btnCloseVid)
        self._vfWidget.setFixedHeight(int(28 * _S))
        self._vfWidget.setVisible(False)
        lay.addWidget(self._vfWidget)

        cam_lbl = QtWidgets.QLabel("Or")
        cam_lbl.setAlignment(QtCore.Qt.AlignCenter)
        cam_lbl.setObjectName("sliderLabel")
        lay.addWidget(cam_lbl)

        cam_row = QtWidgets.QHBoxLayout()
        cam_row.setContentsMargins(0, 0, 0, 0)
        cam_row.setSpacing(int(6 * _S))
        self.camCombo = _CenteredComboBox()
        self.camCombo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.camCombo.setMinimumContentsLength(0)
        
        self.camCombo.setObjectName("loadVideoCombo")
        self.camCombo.setFixedHeight(int(28 * _S))
        self.camCombo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.camCombo.setIconSize(QtCore.QSize(int(13 * _S), int(13 * _S)))
        self.camCombo.addItem(_fa("video", _T, int(13 * _S)), "Choose device", {"type": "none"})
        
        cam_row.addWidget(self.camCombo, stretch=1)

        lay.addLayout(cam_row)
        lay.addSpacing(int(6 * _S))

        # MODELS
        lay.addWidget(self._section("MODELS"))

        self.btnToggleAllModels = QtWidgets.QPushButton("Enable / Disable All")
        self.btnToggleAllModels.setFixedHeight(int(26 * _S))
        self.btnToggleAllModels.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnToggleAllModels.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self.btnToggleAllModels)
        lay.addSpacing(int(4 * _S))

        self._mtog: Dict[int, QtWidgets.QCheckBox] = {}
        self._mcolor_dot: Dict[int, QtWidgets.QPushButton] = {}

        # Single unified picker — every class id gets the same model file.
        # The dicts below are kept for legacy code paths but only ever hold
        # the sentinel ``_UNIFIED_KEY`` entry.
        self._mcombo: Dict[int, _AlignedComboBox] = {}
        self._mconv: Dict[int, QtWidgets.QPushButton] = {}
        self._mbrowse: Dict[int, QtWidgets.QPushButton] = {}
        self._mfmt_label: Dict[int, QtWidgets.QLabel] = {}  # no-op compat

        _bisz = int(11 * _S)  # small icon size
        _bsq = int(22 * _S)   # small button square

        mcard = QtWidgets.QWidget(); mcard.setObjectName("modelCard")
        ml = QtWidgets.QVBoxLayout(mcard)
        ml.setContentsMargins(0, int(4*_S), 0, int(4*_S))
        ml.setSpacing(int(6 * _S))

        # ── Unified model picker (one per app, regardless of class) ──────
        u_lbl = QtWidgets.QLabel("Unified Detector")
        u_lbl.setObjectName("sliderLabel")
        ml.addWidget(u_lbl)

        u_row = QtWidgets.QHBoxLayout()
        u_row.setContentsMargins(0, 0, 0, 0)
        u_row.setSpacing(int(6 * _S))

        u_combo = _AlignedComboBox()
        u_combo.setObjectName("modelCombo")
        u_combo.setMinimumHeight(int(24 * _S))
        u_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed)
        u_combo.view().setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        u_combo.view().setTextElideMode(QtCore.Qt.ElideRight)
        u_combo.view().setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        u_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        u_combo.setMinimumContentsLength(0)
        self._mcombo[_UNIFIED_KEY] = u_combo
        u_row.addWidget(u_combo, stretch=1)

        u_browse = QtWidgets.QPushButton()
        u_browse.setObjectName("iconBtn")
        u_browse.setIcon(_fa("folder-open", _TD, _bisz))
        u_browse.setIconSize(QtCore.QSize(_bisz, _bisz))
        u_browse.setFixedSize(_bsq, _bsq)
        u_browse.setToolTip("Browse for unified detector model")
        u_browse.setFlat(True)
        u_browse.setFocusPolicy(QtCore.Qt.NoFocus)
        u_browse.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._mbrowse[_UNIFIED_KEY] = u_browse
        u_row.addWidget(u_browse, 0, QtCore.Qt.AlignVCenter)

        u_conv = QtWidgets.QPushButton()
        u_conv.setObjectName("iconBtn")
        u_conv.setIcon(_fa("bolt", _TD, _bisz))
        u_conv.setIconSize(QtCore.QSize(_bisz, _bisz))
        u_conv.setFixedSize(_bsq, _bsq)
        u_conv.setToolTip("Optimise unified detector")
        u_conv.setFlat(True)
        u_conv.setFocusPolicy(QtCore.Qt.NoFocus)
        u_conv.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        u_conv.setVisible(False)
        self._mconv[_UNIFIED_KEY] = u_conv
        u_row.addWidget(u_conv, 0, QtCore.Qt.AlignVCenter)

        ml.addLayout(u_row)
        ml.addSpacing(int(4 * _S))

        # ── Per-class detection toggles ─────────────────────────────────
        # The single unified detector emits every class; these checkboxes
        # filter which classes the pipeline acts on (visibility + stats).
        cls_lbl = QtWidgets.QLabel("Classes")
        cls_lbl.setObjectName("sliderLabel")
        ml.addWidget(cls_lbl)

        for cid in TARGET_CLASS_IDS:
            name = CLASS_NAMES[cid]

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(int(6 * _S))

            dot = QtWidgets.QPushButton()
            dot.setObjectName("colorDot")
            _dsz = int(10 * _S)
            dot.setFixedSize(_dsz, _dsz)
            clr = self._to_qcolor_bgr(self._state.get_class_color(cid))
            dot.setStyleSheet(
                f"background:{clr.name()};border:none;"
                f"border-radius:{_dsz // 2}px;"
                f"min-width:{_dsz}px;max-width:{_dsz}px;"
                f"min-height:{_dsz}px;max-height:{_dsz}px;")
            dot.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            dot.setToolTip("Change colour")
            dot.setFlat(True)
            dot.setFocusPolicy(QtCore.Qt.NoFocus)
            self._mcolor_dot[cid] = dot
            row.addWidget(dot, 0, QtCore.Qt.AlignVCenter)

            chk = QtWidgets.QCheckBox(name)
            chk.setChecked(True)
            self._mtog[cid] = chk
            row.addWidget(chk)
            row.addStretch()

            ml.addLayout(row)

        lay.addWidget(mcard)

        # Optimize All button (legacy alias for the unified converter)
        self.btnOptimizeAll = QtWidgets.QPushButton()
        _oisz = int(13 * _S)
        self.btnOptimizeAll.setIcon(_fa("bolt", _T, _oisz))
        self.btnOptimizeAll.setIconSize(QtCore.QSize(_oisz, _oisz))
        self.btnOptimizeAll.setText("  Optimize Unified Detector")
        self.btnOptimizeAll.setFixedHeight(int(28 * _S))
        self.btnOptimizeAll.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnOptimizeAll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btnOptimizeAll.setVisible(False)
        lay.addWidget(self.btnOptimizeAll)

        lay.addSpacing(int(6 * _S))

        # DETECTION
        lay.addWidget(self._section("DETECTION"))
        dcard = QtWidgets.QWidget(); dcard.setObjectName("modelCard")
        dl = QtWidgets.QVBoxLayout(dcard)
        dl.setContentsMargins(0, int(4*_S), 0, int(4*_S))
        dl.setSpacing(int(5 * _S))

        for attr, label, default in [("conf", "Confidence", 25),
                                     ("iou", "IoU", 30)]:
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("sliderLabel")
            dl.addWidget(lbl)
            r = QtWidgets.QHBoxLayout(); r.setSpacing(int(6 * _S))
            r.setContentsMargins(0, 0, 0, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100); slider.setValue(default)
            slider.setMinimumHeight(int(18 * _S))
            slider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            val = QtWidgets.QLabel(f"{default / 100:.2f}")
            val.setObjectName("sliderLabel")
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            val.setFixedWidth(int(28 * _S))
            r.addWidget(slider, stretch=1); r.addWidget(val)
            dl.addLayout(r)
            setattr(self, f"{attr}Slider", slider)
            setattr(self, f"{attr}Label", val)

        perf_lbl = QtWidgets.QLabel("Inference Size")
        perf_lbl.setObjectName("sliderLabel")
        dl.addWidget(perf_lbl)
        self.imgszCombo = _AlignedComboBox()
        self.imgszCombo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.imgszCombo.setMinimumContentsLength(0)
        for size in (256, 320, 480, 640, 960, 1280):
            self.imgszCombo.addItem(str(size), size)
        self.imgszCombo.setCurrentText("480")
        dl.addWidget(self.imgszCombo)

        dev_lbl = QtWidgets.QLabel("Device")
        dev_lbl.setObjectName("sliderLabel")
        dl.addWidget(dev_lbl)
        self.deviceCombo = _AlignedComboBox()
        self.deviceCombo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.deviceCombo.setMinimumContentsLength(0)
        self.deviceCombo.addItem("Auto", "auto")
        self.deviceCombo.addItem("GPU (CUDA)", "cuda")
        self.deviceCombo.addItem("CPU", "cpu")
        dl.addWidget(self.deviceCombo)

        self.chkFp16 = QtWidgets.QCheckBox("Use FP16 instead of FP32")
        self.chkFp16.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.chkFp16.setChecked(False)
        self.chkFp16.setFocusPolicy(QtCore.Qt.NoFocus)
        dl.addWidget(self.chkFp16)

        self.chkOverload = QtWidgets.QCheckBox("Highlight Overloading (>2 riders)")
        self.chkOverload.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.chkOverload.setChecked(True)
        self.chkOverload.setFocusPolicy(QtCore.Qt.NoFocus)
        dl.addWidget(self.chkOverload)

        self.chkDbLogging = QtWidgets.QCheckBox("DB Logging (Supabase)")
        self.chkDbLogging.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.chkDbLogging.setChecked(True)
        self.chkDbLogging.setFocusPolicy(QtCore.Qt.NoFocus)
        dl.addWidget(self.chkDbLogging)

        stride_lbl = QtWidgets.QLabel("Inference Stride")
        stride_lbl.setObjectName("sliderLabel")
        dl.addWidget(stride_lbl)
        sr = QtWidgets.QHBoxLayout(); sr.setSpacing(int(6 * _S))
        sr.setContentsMargins(0, 0, 0, 0)
        self.strideSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.strideSlider.setRange(1, 16)
        self.strideSlider.setValue(3)
        self.strideSlider.setMinimumHeight(int(18 * _S))
        self.strideSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.strideLabel = QtWidgets.QLabel("1")
        self.strideLabel.setObjectName("sliderLabel")
        self.strideLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.strideLabel.setFixedWidth(int(16 * _S))
        sr.addWidget(self.strideSlider, stretch=1)
        sr.addWidget(self.strideLabel)
        dl.addLayout(sr)

        lay.addWidget(dcard)
        lay.addSpacing(int(6 * _S))

        lay.addStretch()
        scroll.setWidget(content); outer.addWidget(scroll)
        return sb

    # ── collapsed (rail) sidebar ──────────────────────────────────────────
    def _build_railbar(self) -> QtWidgets.QWidget:
        """Narrow rail shown when the sidebar is collapsed.

        Top-aligned, horizontally centred. Contains:
          * Hamburger to expand back to the full sidebar.
          * One coloured pill per class (click toggles detection).
        """
        rb = QtWidgets.QWidget()
        rb.setObjectName("sidebar")
        rail_w = int(54 * _S)
        rb.setFixedWidth(rail_w)
        v = QtWidgets.QVBoxLayout(rb)
        v.setContentsMargins(int(6 * _S), int(10 * _S),
                             int(6 * _S), int(10 * _S))
        v.setSpacing(int(8 * _S))
        v.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        # Expand-back button at the top of the rail.
        self.btnRailExpand = self._ibtn("bars", "Expand sidebar")
        v.addWidget(self.btnRailExpand, 0, QtCore.Qt.AlignHCenter)

        # Subtle divider
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setStyleSheet(f"color:{_BD};background:{_BD};max-height:1px;")
        v.addWidget(sep1)

        self._rail_chk: Dict[int, QtWidgets.QCheckBox] = {}
        self._rail_dot: Dict[int, QtWidgets.QPushButton] = {}

        # Per-class "pill" buttons. Coloured circle that doubles as the
        # detection toggle; greyed-out when the class is disabled.
        _pill = int(30 * _S)
        for cid in TARGET_CLASS_IDS:
            name = CLASS_NAMES.get(cid, "?")
            clr = self._to_qcolor_bgr(self._state.get_class_color(cid))

            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            chk.setVisible(False)        # state holder only
            chk.toggled.connect(lambda v, c=cid: self._on_rail_chk_toggled(c, v))
            self._rail_chk[cid] = chk

            dot = QtWidgets.QPushButton(name[:1].upper())
            dot.setObjectName("railPill")
            dot.setFixedSize(_pill, _pill)
            dot.setMinimumSize(_pill, _pill)
            dot.setMaximumSize(_pill, _pill)
            dot.setCheckable(True)
            dot.setChecked(True)
            dot.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            dot.setFocusPolicy(QtCore.Qt.NoFocus)
            dot.setFlat(True)
            dot.setToolTip(f"{name} — click to toggle, right-click for colour")
            dot.setStyleSheet(self._rail_pill_qss(clr, _pill))

            def _on_pill_toggled(v, c=cid):
                rail_chk = self._rail_chk.get(c)
                if rail_chk is not None and rail_chk.isChecked() != v:
                    rail_chk.setChecked(v)

            dot.toggled.connect(_on_pill_toggled)
            # Right-click → pick colour, mirroring expanded sidebar.
            dot.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            dot.customContextMenuRequested.connect(
                lambda _pos, c=cid: self._on_pick_model_color(c))

            self._rail_dot[cid] = dot
            v.addWidget(dot, 0, QtCore.Qt.AlignHCenter)

        v.addStretch(1)
        return rb

    def _rail_pill_qss(self, clr: QtGui.QColor, size: int) -> str:
        """Build the QSS for a perfect-circle rail pill at *size* px."""
        r = size // 2
        font_px = int(13 * _S)
        return (
            "QPushButton#railPill{"
            f"background:{clr.name()};color:#0b0d12;"
            "border:none;padding:0;margin:0;"
            f"min-width:{size}px;max-width:{size}px;"
            f"min-height:{size}px;max-height:{size}px;"
            f"border-radius:{r}px;"
            f"font-weight:700;font-size:{font_px}px;"
            "text-align:center;"
            "}"
            "QPushButton#railPill:!checked{"
            f"background:{_BG};color:{_TD};"
            f"border:1px solid {_BD};"
            "}"
            "QPushButton#railPill:hover{"
            f"border:1px solid {_TH};"
            "}"
        )

    def _on_rail_chk_toggled(self, cid: int, value: bool) -> None:
        """Mirror rail checkbox state into the full sidebar checkbox."""
        chk = self._mtog.get(cid)
        if chk is not None and chk.isChecked() != value:
            chk.setChecked(value)
        else:
            self._state.set_model_enabled(cid, value)

    # ── console / logs panel ──────────────────────────────────────────────
    def _build_console_panel(self) -> QtWidgets.QWidget:
        """Right-side terminal-style log panel with Console / Errors tabs."""
        panel = QtWidgets.QWidget()
        panel.setObjectName("consolePanel")
        panel.setMinimumWidth(int(100 * _S))
        panel.setMaximumWidth(int(640 * _S))
        panel.setStyleSheet(
            f"#consolePanel{{background:{_BG_SB};"
            f"border-left:1px solid {_BD};}}"
        )

        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar
        header = QtWidgets.QWidget()
        header.setFixedHeight(int(32 * _S))
        header.setStyleSheet(
            f"background:{_BG};border-bottom:1px solid {_BD};")
        h = QtWidgets.QHBoxLayout(header)
        h.setContentsMargins(int(10 * _S), 0, int(6 * _S), 0)
        h.setSpacing(int(6 * _S))
        title = QtWidgets.QLabel("Console")
        title.setStyleSheet(
            f"color:{_TH};font-weight:600;font-size:{int(11 * _S)}px;")
        h.addWidget(title)
        h.addStretch()
        self.btnClearConsole = self._ibtn("trash", "Clear logs", sz=14)
        self.btnCloseConsole = self._ibtn("times", "Close console", sz=14)
        h.addWidget(self.btnClearConsole)
        h.addWidget(self.btnCloseConsole)
        outer.addWidget(header)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("consoleTabs")
        tabs.setDocumentMode(True)
        tabs.setStyleSheet(
            f"QTabWidget::pane{{border:none;background:{_BG_SB};}}"
            f"QTabBar::tab{{background:transparent;color:{_T};"
            f"padding:{int(6*_S)}px {int(14*_S)}px;"
            f"font-size:{int(10*_S)}px;border:none;"
            f"border-bottom:2px solid transparent;}}"
            f"QTabBar::tab:selected{{color:{_W};"
            f"border-bottom:2px solid {_W};background:{_BG};}}"
            f"QTabBar::tab:hover{{color:{_TH};background:{_HOV};}}"
        )

        mono_fam = "Consolas, 'Cascadia Mono', 'Fira Code', monospace"

        def _mk_view() -> QtWidgets.QPlainTextEdit:
            v = QtWidgets.QPlainTextEdit()
            v.setReadOnly(True)
            v.setMaximumBlockCount(2000)  # cap memory
            v.setFrameShape(QtWidgets.QFrame.NoFrame)
            v.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            v.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
            v.setStyleSheet(
                f"QPlainTextEdit{{background:{_BG_SB};color:{_T};"
                f"font-family:{mono_fam};font-size:{int(8*_S)}px;"
                f"selection-background-color:{_PRS};"
                f"padding:{int(4*_S)}px {int(8*_S)}px;"
                f"line-height:1.4;}}"
                f"QScrollBar:vertical{{background:{_BG_SB};width:{int(6*_S)}px;}}"
                f"QScrollBar::handle:vertical{{background:#2a2a2a;"
                f"border-radius:{int(3*_S)}px;min-height:24px;}}"
                f"QScrollBar::handle:vertical:hover{{background:#3a3a3a;}}"
                f"QScrollBar::add-line:vertical,"
                f"QScrollBar::sub-line:vertical{{height:0;}}"
            )
            return v

        self._consoleView = _mk_view()
        self._errorsView = _mk_view()

        tabs.addTab(self._consoleView, "Console")
        tabs.addTab(self._errorsView, "Errors")
        outer.addWidget(tabs, stretch=1)

        self._consoleTabs = tabs
        return panel

    def _toggle_console(self) -> None:
        if self._consolePanel.isVisible():
            # Remember the width so we can restore it next time
            sizes = self._splitter.sizes()
            if len(sizes) >= 3 and sizes[2] > 0:
                self._console_last_w = sizes[2]
            self._consolePanel.hide()
            if len(sizes) >= 2:
                self._splitter.setSizes(
                    [sizes[0], sizes[1] + sum(sizes[2:]), 0])
        else:
            self._consolePanel.show()
            target = int(getattr(self, "_console_last_w", 0)
                         or self._console_default_w)
            sizes = self._splitter.sizes()
            if len(sizes) >= 2 and sizes[1] > target + int(200 * _S):
                self._splitter.setSizes(
                    [sizes[0], sizes[1] - target, target])
        QtCore.QTimer.singleShot(0, self._repos)

    def _log(self, message: str, level: str = "info") -> None:
        """Append a line to the console (and to the Errors tab if level=error)."""
        if not message:
            return
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        view = getattr(self, "_consoleView", None)
        if view is not None:
            view.appendPlainText(line)
        if level == "error":
            err = getattr(self, "_errorsView", None)
            if err is not None:
                err.appendPlainText(line)
            # Briefly flash status bar with the error text
            try:
                self.statusLabel.setText(f"⚠ {message}")
            except Exception:
                pass

    def _log_error(self, message: str) -> None:
        self._log(message, level="error")

    def _on_clear_console(self) -> None:
        idx = self._consoleTabs.currentIndex() if hasattr(self, "_consoleTabs") else 0
        view = self._errorsView if idx == 1 else self._consoleView
        view.clear()

    def _on_violation_event(self, payload: dict) -> None:
        """Tracker → console + Supabase logger.

        Only NON-COMPLIANT events are routed (helmet missing / improper
        footwear / overload).  Throttling lives in supabase_logger.submit().
        """
        vtype = str(payload.get("violation_type", "")).strip()
        if not vtype:
            return
        # Skip overload events for Supabase logging — user only asked for
        # footwear / helmet violations to be uploaded.
        log_to_db = vtype in ("no_helmet", "improper_footwear")

        # Always log to the in-app console
        label = _VIOLATION_LABELS.get(vtype, vtype.upper())
        moto = payload.get("motorcycle_id")
        rider = payload.get("rider_id")
        rc = payload.get("rider_count")
        conf = payload.get("confidence")
        ts_ms = payload.get("timestamp_ms")
        bits = [label]
        if moto is not None:
            bits.append(f"moto#{moto}")
        if rider is not None:
            bits.append(f"rider#{rider}")
        if rc is not None:
            bits.append(f"riders={rc}")
        if isinstance(conf, (int, float)):
            bits.append(f"conf={conf:.2f}")
        if isinstance(ts_ms, (int, float)) and ts_ms >= 0:
            total_s = int(ts_ms / 1000)
            m, s = divmod(total_s, 60)
            frac = int((ts_ms / 1000 - total_s) * 10)
            bits.append(f"@{m}:{s:02d}.{frac}")
        msg = " ".join(bits)

        # Dedupe at GUI level too — avoids spamming console every frame
        # while the same offence is on screen.
        key = f"{vtype}:{moto}:{rider}"
        now = time.monotonic()
        if not hasattr(self, "_violation_seen"):
            self._violation_seen: Dict[str, float] = {}
        if now - self._violation_seen.get(key, 0.0) < 8.0:
            return
        self._violation_seen[key] = now
        # GC
        if len(self._violation_seen) > 512:
            cutoff = now - 32.0
            for k in [k for k, t in self._violation_seen.items() if t < cutoff]:
                self._violation_seen.pop(k, None)

        self._log(msg)

        if log_to_db and _sblog.is_enabled():
            try:
                _sblog.submit(_sblog.ViolationEvent(
                    violation_type=vtype,
                    motorcycle_id=int(moto) if isinstance(moto, int) else None,
                    rider_id=int(rider) if isinstance(rider, int) else None,
                    rider_count=int(rc) if isinstance(rc, int) else None,
                    confidence=float(conf) if isinstance(conf, (int, float)) else None,
                    source=self._current_source_label(),
                    video_timestamp=float(ts_ms) if isinstance(ts_ms, (int, float)) else None,
                ))
            except Exception as e:
                self._log_error(f"Supabase submit error: {e}")

    def _current_source_label(self) -> Optional[str]:
        try:
            path, cam = self._state.get_source()
            if path:
                return os.path.basename(path)
            if cam is not None:
                return f"camera:{cam}"
        except Exception:
            pass
        return None

    def _sync_rail_from_sidebar(self) -> None:
        """Push sidebar checkbox / colour state onto the rail pills."""
        if not getattr(self, "_rail_chk", None):
            return
        _pill = int(30 * _S)
        for cid in TARGET_CLASS_IDS:
            src = self._mtog.get(cid)
            dst = self._rail_chk.get(cid)
            if src is not None and dst is not None and dst.isChecked() != src.isChecked():
                dst.blockSignals(True)
                dst.setChecked(src.isChecked())
                dst.blockSignals(False)
            dot = self._rail_dot.get(cid)
            if dot is not None:
                clr = self._to_qcolor_bgr(self._state.get_class_color(cid))
                checked = src.isChecked() if src is not None else True
                if dot.isChecked() != checked:
                    dot.blockSignals(True)
                    dot.setChecked(checked)
                    dot.blockSignals(False)
                dot.setStyleSheet(self._rail_pill_qss(clr, _pill))

    def _toggle_sidebar_collapsed(self) -> None:
        self._sidebar_collapsed = not getattr(self, "_sidebar_collapsed", False)
        if self._sidebar_collapsed:
            self._sidebarStack.setCurrentIndex(1)
            rail_w = int(54 * _S)
            self._sidebarStack.setMinimumWidth(rail_w)
            self._sidebarStack.setMaximumWidth(rail_w)
            self._splitter.setSizes(
                [rail_w, max(self.width() - rail_w, int(400 * _S))])
            self._sync_rail_from_sidebar()
        else:
            self._sidebarStack.setCurrentIndex(0)
            self._sidebarStack.setMinimumWidth(int(180 * _S))
            self._sidebarStack.setMaximumWidth(int(300 * _S))
            self._splitter.setSizes(
                [int(260 * _S), max(self.width() - int(260 * _S), int(400 * _S))])
        QtCore.QTimer.singleShot(0, self._repos)

    def _update_model_format_labels(self) -> None:
        """No-op kept for legacy callers; sidebar no longer shows per-class\n        format labels under the unified-detector layout."""
        return

    # ── overlay positioning ───────────────────────────────────────────────

    def _repos(self) -> None:
        cw, ch = self._vc.width(), self._vc.height()
        self._overlay.adjustSize()
        oh = max(self._overlay.sizeHint().height(),
                 self._overlay.layout().minimumSize().height(),
                 int(60 * _S))  # fallback minimum
        self._overlay.setGeometry(0, ch - oh, cw, oh)
        self._overlay.raise_()

    def eventFilter(self, obj, event):
        etype = event.type()
        # video container resize → reposition overlay + rescale pixmap
        if obj is self._vc and etype == QtCore.QEvent.Resize:
            self._repos()
            if not self._last_pixmap.isNull():
                mode = self._video_scale_mode()
                self.videoDisplay.setPixmap(self._last_pixmap.scaled(
                    self.videoDisplay.size(), mode,
                    QtCore.Qt.FastTransformation))
        # fullscreen: mouse movement on video/overlay → show controls
        if self._is_fs and etype == QtCore.QEvent.MouseMove:
            if obj in (self._vc, self.videoDisplay, self._overlay):
                self._fs_show_controls()
        # windowed HUD auto-hide: when not pinned, mouse movement on the
        # video area shows the overlay and restarts the 2 s idle timer.
        if (not self._is_fs) and (not getattr(self, "_hud_locked", True)):
            if obj in (self._vc, self.videoDisplay, self._overlay):
                if etype in (QtCore.QEvent.MouseMove, QtCore.QEvent.Enter):
                    self._hud_show_temp()
                elif etype == QtCore.QEvent.Leave and obj is self._vc:
                    # cursor left the video viewport entirely → start
                    # the countdown immediately so the HUD fades fast.
                    self._hud_idle_timer.start(500)
        # seek slider hover for preview
        if obj is self.seekSlider:
            if etype == QtCore.QEvent.MouseMove:
                self._on_seek_hover(event.pos().x())
            elif etype == QtCore.QEvent.Leave:
                self._seekPrev.hide()

        if obj is self.videoDisplay and etype == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton and self._has_active_source():
                self._on_play_pause()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._repos()

    def showEvent(self, ev):
        super().showEvent(ev)
        QtCore.QTimer.singleShot(0, self._repos)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Escape:
            if self._is_fs:
                self._exit_fullscreen()
                return
        super().keyPressEvent(ev)

    def _setup_shortcuts(self) -> None:
        self._shortcuts: list[QtWidgets.QShortcut] = []

        def _bind(seq: str, fn) -> None:
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

        _bind("Space", self._on_play_pause)
        _bind("K", self._on_play_pause)
        _bind("J", lambda: self._skip(-10))
        _bind("L", lambda: self._skip(10))
        _bind("Left", lambda: self._skip(-5))
        _bind("Right", lambda: self._skip(5))
        _bind("F", self._toggle_fullscreen)
        _bind("Escape", lambda: self._is_fs and self._exit_fullscreen())

    def _set_fs_button(self, icon: str, tooltip: str) -> None:
        self.btnFS.setIcon(_fa(icon, _W, _FS_ICON_SIZE))
        self.btnFS.setIconSize(QtCore.QSize(_FS_ICON_SIZE, _FS_ICON_SIZE))
        self.btnFS.setToolTip(tooltip)

    def _set_video_ui_state(self, loaded: bool) -> None:
        self.btnLoadVideo.setVisible(not loaded)
        self._vfWidget.setVisible(loaded)

    def _set_source_label(self, text: str, tooltip: str) -> None:
        self.lblVidName.setText(text)
        self.lblVidName.setToolTip(tooltip)
        self._set_video_ui_state(True)

    def _set_camera_combo_none(self) -> None:
        if self.camCombo.count() <= 0:
            return
        self.camCombo.blockSignals(True)
        self.camCombo.setCurrentIndex(0)
        self.camCombo.blockSignals(False)

    def _refresh_camera_devices(self) -> None:
        if self._cam_scan and self._cam_scan.isRunning():
            return
        # If a camera is currently in use by the pipeline, skip the scan —
        # probing the same device a second time on Windows often steals it
        # away from the running grabber.
        _path, active_cam = self._state.get_source()
        if active_cam is not None:
            return
        self.camCombo.setEnabled(False)
        self.btnRefreshCams.setEnabled(False)
        self.btnRefreshCams.setIcon(_fa("spinner", "#FFC107", int(11 * _S)))

        self._cam_scan = _CameraScanThread(max_index=4, parent=self)
        self._cam_scan.completed.connect(self._on_camera_scan_completed)
        self._cam_scan.failed.connect(self._on_camera_scan_failed)
        self._cam_scan.start()

    def _on_camera_scan_completed(self, devices: list) -> None:
        prev_data = self.camCombo.currentData() or {}
        prev_key = None
        if isinstance(prev_data, dict):
            prev_cam = prev_data.get("camera") if prev_data.get("type") == "camera" else None
            if isinstance(prev_cam, dict):
                prev_key = prev_cam.get("key")
        _path, active_cam = self._state.get_source()
        active_key = active_cam.get("key") if isinstance(active_cam, dict) else None

        self.camCombo.blockSignals(True)
        self.camCombo.clear()
        cam_isz = int(13 * _S)
        self.camCombo.addItem(_fa("video", _T, cam_isz), "Choose device", {"type": "none"})

        selected_idx = 0
        seen_keys = set()
        for i, dev in enumerate(devices, start=1):
            if not isinstance(dev, dict):
                continue
            key = str(dev.get("key") or "")
            label = str(dev.get("label") or "Camera")
            if not key:
                continue
            self.camCombo.addItem(_fa("video", _T, cam_isz), label, {"type": "camera", "camera": dev})
            seen_keys.add(key)
            if prev_key is not None and key == str(prev_key):
                selected_idx = i

        if isinstance(active_cam, dict) and active_key and active_key not in seen_keys:
            fallback_label = str(active_cam.get("label") or "Active Camera") + " (in use)"
            self.camCombo.addItem(
                _fa("video", _T, cam_isz),
                fallback_label,
                {
                    "type": "camera",
                    "camera": {
                        **active_cam,
                        "label": fallback_label,
                    },
                },
            )
            if prev_key is None or str(prev_key) == str(active_key):
                selected_idx = self.camCombo.count() - 1

        self.camCombo.setCurrentIndex(selected_idx)
        self.camCombo.blockSignals(False)

        self.camCombo.setEnabled(True)
        self.btnRefreshCams.setEnabled(True)
        self.btnRefreshCams.setIcon(_fa("sync-alt", _TD, int(11 * _S)))

    def _on_camera_scan_failed(self, message: str) -> None:
        self.camCombo.setEnabled(True)
        self.btnRefreshCams.setEnabled(True)
        self.btnRefreshCams.setIcon(_fa("sync-alt", _TD, int(11 * _S)))
        self._set_status(f"Camera scan failed: {message}")

    def _on_camera_combo_changed(self) -> None:
        data = self.camCombo.currentData() or {}
        kind = data.get("type") if isinstance(data, dict) else "none"
        if kind == "none":
            _path, cam = self._state.get_source()
            if cam is not None:
                self._on_close_video()
            return

        if kind == "camera":
            cam = data.get("camera") if isinstance(data, dict) else None
            if isinstance(cam, dict):
                _path, active_cam = self._state.get_source()
                if isinstance(active_cam, dict) and active_cam.get("key") == cam.get("key"):
                    return
                label = str(cam.get("label") or "Camera")
                self._open_camera(cam, label)

    # ── fullscreen ────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self._is_fs:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        self._is_fs = True
        self._sidebarStack.hide()
        self._topBar.hide()
        self.statusLabel.hide()
        self._set_fs_button("compress", "Exit Fullscreen")
        self.showFullScreen()
        self._overlay.show()
        self._fs_hide.start(3000)
        QtCore.QTimer.singleShot(0, self._repos)

    def _exit_fullscreen(self) -> None:
        self._is_fs = False
        self._fs_hide.stop()
        self._sidebarStack.show()
        self._topBar.show()
        self.statusLabel.show()
        self._set_fs_button("expand", "Fullscreen")
        self._overlay.show()
        self.videoDisplay.setCursor(QtCore.Qt.ArrowCursor)
        self.showNormal()
        QtCore.QTimer.singleShot(0, self._repos)

    def _fs_show_controls(self) -> None:
        self._overlay.show()
        self._repos()
        self.videoDisplay.setCursor(QtCore.Qt.ArrowCursor)
        self._fs_hide.start(3000)

    def _fs_hide_overlay(self) -> None:
        if self._is_fs:
            self._overlay.hide()
            self._seekPrev.hide()
            self.videoDisplay.setCursor(QtCore.Qt.BlankCursor)

    # ── windowed HUD pin / auto-hide ──────────────────────────────────────
    def _toggle_hud_lock(self) -> None:
        self._hud_locked = not getattr(self, "_hud_locked", True)
        self._apply_hud_lock_state()

    def _apply_hud_lock_state(self) -> None:
        locked = bool(getattr(self, "_hud_locked", True))
        # Update icon + tooltip
        if hasattr(self, "btnHudLock"):
            icon = _fa("eye" if locked else "eye-slash", _T, int(15 * _S))
            self.btnHudLock.setIcon(icon)
            self.btnHudLock.setToolTip(
                "HUD pinned (click to auto-hide)" if locked
                else "HUD auto-hides (click to pin)")
        # Apply effect
        self._hud_idle_timer.stop()
        if locked or self._is_fs:
            self._overlay.show()
            self._repos()
        else:
            # Show briefly so the user notices the change, then fade.
            self._hud_show_temp()

    def _hud_show_temp(self) -> None:
        if self._is_fs:
            return
        self._overlay.show()
        self._repos()
        self._hud_idle_timer.start(2000)

    def _hud_auto_hide(self) -> None:
        if self._is_fs or getattr(self, "_hud_locked", True):
            return
        # If the cursor is still over the video container, don't hide yet.
        try:
            pos = self._vc.mapFromGlobal(QtGui.QCursor.pos())
            if self._vc.rect().contains(pos):
                self._hud_idle_timer.start(2000)
                return
        except Exception:
            pass
        self._overlay.hide()
        self._seekPrev.hide()

    # ── screenshot ────────────────────────────────────────────────────────

    def _take_screenshot(self) -> None:
        pm = self.videoDisplay.pixmap()
        if pm and not pm.isNull():
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Screenshot", "screenshot.png",
                "PNG (*.png);;JPEG (*.jpg)")
            if p:
                pm.save(p)
                self._set_status(f"Saved: {os.path.basename(p)}")
        else:
            self._set_status("No frame to capture")

    # ── signals ───────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.btnLoadVideo.clicked.connect(self._on_load_video)
        self.btnCloseVid.clicked.connect(self._on_close_video)
        self.camCombo.currentIndexChanged.connect(self._on_camera_combo_changed)
        self.btnRefreshCams.clicked.connect(self._refresh_camera_devices)
        for cid, combo in self._mcombo.items():
            combo.currentIndexChanged.connect(
                lambda idx, c=cid: self._on_model_combo_changed(c))
        for cid, conv in self._mconv.items():
            conv.clicked.connect(
                lambda _=False, c=cid: self._on_convert_model(c))
        for cid, bb in self._mbrowse.items():
            bb.clicked.connect(
                lambda _=False, c=cid: self._on_browse_model(c))
        for cid, dot in self._mcolor_dot.items():
            dot.clicked.connect(
                lambda _=False, c=cid: self._on_pick_model_color(c))
        self.btnOptimizeAll.clicked.connect(self._on_convert_all)
        self.videoDisplay.filesDropped.connect(self._on_files_dropped)
        self.btnPlay.clicked.connect(self._on_play_pause)
        self.btnStepFwd.clicked.connect(self._step_forward)
        self.btnRew.clicked.connect(lambda: self._skip(-10))
        self.btnFwd.clicked.connect(lambda: self._skip(10))
        self.btnShot.clicked.connect(self._take_screenshot)
        self.btnFS.clicked.connect(self._toggle_fullscreen)
        self.seekSlider.sliderPressed.connect(
            lambda: setattr(self, "_user_seeking", True))
        self.seekSlider.sliderReleased.connect(self._on_seek_released)
        self.seekSlider.sliderMoved.connect(self._on_seek_moved)
        self.spdSlider.valueChanged.connect(self._on_speed_changed)
        self.btnToggleAllModels.clicked.connect(self._on_toggle_all_models)
        for cid, chk in self._mtog.items():
            chk.toggled.connect(
                lambda v, c=cid: self._on_sidebar_chk_toggled(c, v))
        self.confSlider.valueChanged.connect(self._on_conf)
        self.iouSlider.valueChanged.connect(self._on_iou)
        self.imgszCombo.currentIndexChanged.connect(self._on_imgsz_changed)
        self.deviceCombo.currentIndexChanged.connect(self._on_device_changed)
        self.chkFp16.toggled.connect(self._on_fp16_toggled)
        self.chkOverload.toggled.connect(self._on_overload_toggled)
        self.chkDbLogging.toggled.connect(self._on_db_logging_toggled)
        self.strideSlider.valueChanged.connect(self._on_stride_changed)
        self.btnHamburger.clicked.connect(self._toggle_sidebar_collapsed)
        self.btnRailExpand.clicked.connect(self._toggle_sidebar_collapsed)
        self.btnHudLock.clicked.connect(self._toggle_hud_lock)
        self.btnLogs.clicked.connect(self._toggle_console)
        self.btnClearConsole.clicked.connect(self._on_clear_console)
        self.btnCloseConsole.clicked.connect(self._toggle_console)

    def _on_sidebar_chk_toggled(self, cid: int, value: bool) -> None:
        self._state.set_model_enabled(cid, value)
        rail = getattr(self, "_rail_chk", {}).get(cid) if hasattr(self, "_rail_chk") else None
        if rail is not None and rail.isChecked() != value:
            rail.blockSignals(True)
            rail.setChecked(value)
            rail.blockSignals(False)

    def _on_pick_model_color(self, cid: int) -> None:
        current = self._to_qcolor_bgr(self._state.get_class_color(cid))
        picked = QtWidgets.QColorDialog.getColor(
            current,
            self,
            f"Choose color: {CLASS_NAMES.get(cid, str(cid))}",
        )
        if not picked.isValid():
            return
        self._state.set_class_color(cid, self._to_bgr_tuple(picked))
        # update sidebar colour dot
        dot = self._mcolor_dot.get(cid)
        if dot:
            _dsz = int(10 * _S)
            dot.setStyleSheet(
                f"background:{picked.name()};border:none;"
                f"border-radius:{_dsz // 2}px;"
                f"min-width:{_dsz}px;max-width:{_dsz}px;"
                f"min-height:{_dsz}px;max-height:{_dsz}px;")
        # update rail (collapsed) dot
        rail_dot = getattr(self, "_rail_dot", {}).get(cid) if hasattr(self, "_rail_dot") else None
        if rail_dot:
            _dsz = int(10 * _S)
            rail_dot.setStyleSheet(
                f"background:{picked.name()};border:none;"
                f"border-radius:{_dsz // 2}px;"
                f"min-width:{_dsz}px;max-width:{_dsz}px;"
                f"min-height:{_dsz}px;max-height:{_dsz}px;")

    def _on_toggle_all_models(self) -> None:
        """Toggle all model-class checkboxes on/off together."""
        if not self._mtog:
            return
        all_enabled = all(chk.isChecked() for chk in self._mtog.values())
        target = not all_enabled
        for chk in self._mtog.values():
            chk.setChecked(target)
        self._set_status(
            "All model classes enabled" if target
            else "All model classes disabled")

    def _on_auto_models(self) -> None:
        combo = self._mcombo.get(_UNIFIED_KEY)
        if combo is not None and combo.count() > 0:
            combo.setCurrentIndex(0)
        self._set_status("Unified detector set to Auto")

    def _set_all_models_format(self, fmt: str, label: str) -> None:
        """Set the unified detector combobox to the requested runtime format."""
        combo = self._mcombo.get(_UNIFIED_KEY)
        if combo is None:
            self._set_status("No unified detector picker available")
            return
        target_idx = -1
        for idx in range(combo.count()):
            data = combo.itemData(idx) or {}
            if data.get("type") in ("variant", "custom") and data.get("format") == fmt:
                target_idx = idx
                break
        if target_idx >= 0:
            if combo.currentIndex() != target_idx:
                combo.setCurrentIndex(target_idx)
            self._set_status(f"Unified detector set to {label}")
        else:
            self._set_status(f"{label} format not available for unified detector")

    def _step_forward(self) -> None:
        path, _cam = self._state.get_source()
        if path:
            self._skip(1 / max(self._video_fps, 1))

    # ── auto-load models ──────────────────────────────────────────────────

    def _auto_load_models(self) -> None:
        # Always purge stray .onnx intermediates left over from previous
        # optimization runs — TensorRT is the canonical GPU artifact.
        try:
            cleanup_onnx_artifacts()
        except Exception:
            pass
        self._model_groups = discover_models()
        # Single unified picker — populate it once and report whether the
        # detector was found.
        self._populate_model_combo(_UNIFIED_KEY)
        probe_cid = next(iter(TARGET_CLASS_IDS))
        probe = self._model_groups.get(probe_cid)
        loaded = ["Unified Detector"] if (probe and probe.variants) else []
        use_cuda = self._has_cuda()
        default_imgsz = 640 if use_cuda else 480
        default_stride = 1 if use_cuda else 2
        default_batch = 4 if use_cuda else 2
        self._state.set_imgsz(default_imgsz)
        self._state.set_device("cuda" if use_cuda else "auto")
        self._state.set_use_fp16(use_cuda)
        self._state.set_inference_stride(default_stride)
        self._state.set_inference_batch_size(default_batch)
        self._state.set_conf(0.25)
        self._state.set_iou(0.30)

        # Reflect new defaults in the sidebar widgets
        self.confSlider.blockSignals(True)
        self.confSlider.setValue(25)
        self.confSlider.blockSignals(False)
        self.confLabel.setText("0.25")
        self.iouSlider.blockSignals(True)
        self.iouSlider.setValue(30)
        self.iouSlider.blockSignals(False)
        self.iouLabel.setText("0.30")
        self.imgszCombo.setCurrentText(str(default_imgsz))
        self.strideSlider.setValue(default_stride)
        self.strideLabel.setText(str(default_stride))
        self.deviceCombo.setCurrentIndex(1 if use_cuda else 0)
        self.chkFp16.setChecked(use_cuda)
        self._set_status(
            f"Auto-loaded: {', '.join(loaded)}" if loaded
            else "No default models found \u2014 load via sidebar")
        QtCore.QTimer.singleShot(0, self._update_convert_buttons)
        self._update_model_format_labels()

    # ── video / model loading ─────────────────────────────────────────────

    def _on_load_video(self) -> None:
        source_dir = os.path.join(_ROOT, "source")
        start_dir = source_dir if os.path.isdir(source_dir) else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All (*)")
        if p:
            self._open_video(p)

    # ── model selector helpers ────────────────────────────────────────────

    def _get_effective_device(self) -> str:
        """Return 'cuda' or 'cpu' based on current combo selection."""
        raw = str(self.deviceCombo.currentData() or "auto").lower()
        if raw == "cpu":
            return "cpu"
        if raw in ("cuda", "auto") and self._has_cuda():
            return "cuda"
        return "cpu"

    def _mirror_unified_path(self, path: Optional[str]) -> None:
        """Set ``state.model_paths[cid] = path`` for every target class id.

        With a single unified detector, every class id resolves to the
        same physical model file.
        """
        for tcid in TARGET_CLASS_IDS:
            self._state.model_paths[tcid] = path

    def _populate_model_combo(self, cid: int = _UNIFIED_KEY) -> None:
        """Fill the unified model combobox with Auto + discovered variants."""
        combo = self._mcombo[_UNIFIED_KEY]
        combo.blockSignals(True)
        combo.clear()

        combo.addItem("Auto (best for device)", {"type": "auto"})

        # All groups share the same variant list under the unified scheme;
        # any cid works as a probe.
        probe_cid = next(iter(TARGET_CLASS_IDS))
        group = self._model_groups.get(probe_cid)
        if group and group.variants:
            for v in group.variants:
                combo.addItem(v.display_name, {
                    "type": "variant", "path": v.path, "format": v.format,
                })

        combo.setCurrentIndex(0)
        combo.blockSignals(False)

        self._prev_combo_idx[_UNIFIED_KEY] = 0
        self._resolve_auto_model(_UNIFIED_KEY)

    def _resolve_auto_model(self, cid: int = _UNIFIED_KEY) -> None:
        """Resolve the 'Auto' selection to the best variant for the device."""
        probe_cid = next(iter(TARGET_CLASS_IDS))
        group = self._model_groups.get(probe_cid)
        combo = self._mcombo[_UNIFIED_KEY]
        if not group or not group.variants:
            self._mirror_unified_path(None)
            combo.setToolTip("No models found")
            return
        device = self._get_effective_device()
        variant = group.best_for_device(device)
        if variant:
            self._mirror_unified_path(variant.path)
            combo.setToolTip(f"Using: {variant.display_name}")
        else:
            self._mirror_unified_path(None)
            combo.setToolTip("No compatible model for current device")

    def _on_model_combo_changed(self, cid: int = _UNIFIED_KEY) -> None:
        """Handle unified model combobox selection change."""
        combo = self._mcombo[_UNIFIED_KEY]
        data = combo.currentData()
        if not data:
            return

        kind = data.get("type")

        if kind == "auto":
            self._resolve_auto_model(_UNIFIED_KEY)
        elif kind in ("variant", "custom"):
            self._mirror_unified_path(data["path"])
            combo.setToolTip(data["path"])

        self._prev_combo_idx[_UNIFIED_KEY] = combo.currentIndex()
        self._update_convert_buttons()
        self._update_model_format_labels()

    def _on_browse_model(self, cid: int = _UNIFIED_KEY) -> None:
        """Open a file/folder dialog for browsing the unified model."""
        from utils.model_registry import MODELS_DIR
        start_dir = MODELS_DIR if os.path.isdir(MODELS_DIR) else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Unified Detector", start_dir,
            "Model Files (*.pt *.engine *.onnx);;OpenVINO (*.xml);;All (*)")
        if p:
            # If user picked an .xml inside an openvino dir, use the parent dir
            if p.endswith(".xml"):
                parent = os.path.dirname(p)
                if _detect_format(parent) == "openvino":
                    p = parent
            self._add_custom_model(_UNIFIED_KEY, p)
            return

    def _add_custom_model(self, cid: int, path: str) -> None:
        """Add a user-browsed model to the unified combobox and select it."""
        fmt = _detect_format(path)
        if not fmt:
            QtWidgets.QMessageBox.warning(
                self, "Unsupported Format",
                f"Cannot detect model format:\n{path}")
            return
        label = f"{os.path.basename(path)}  ({_FORMAT_LABEL.get(fmt, fmt)})"
        combo = self._mcombo[_UNIFIED_KEY]
        combo.blockSignals(True)
        insert_idx = combo.count()
        combo.insertItem(insert_idx, label, {
            "type": "custom", "path": path, "format": fmt,
        })
        combo.setCurrentIndex(insert_idx)
        combo.blockSignals(False)
        self._mirror_unified_path(path)
        combo.setToolTip(path)
        self._prev_combo_idx[_UNIFIED_KEY] = insert_idx
        self._update_convert_buttons()
        self._update_model_format_labels()

    def _refresh_model_combos(self) -> None:
        """Re-discover models and repopulate the unified combobox."""
        self._model_groups = discover_models()
        combo = self._mcombo[_UNIFIED_KEY]
        old_data = combo.currentData() or {}
        was_auto = old_data.get("type") == "auto"
        old_path = old_data.get("path")

        self._populate_model_combo(_UNIFIED_KEY)

        if not was_auto and old_path:
            for i in range(combo.count()):
                d = combo.itemData(i) or {}
                if d.get("path") == old_path:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    self._mirror_unified_path(old_path)
                    self._prev_combo_idx[_UNIFIED_KEY] = i
                    break
        self._update_model_format_labels()

    def _set_model(self, cid: int, path: str) -> None:
        """Programmatic model set (e.g. from drag-and-drop)."""
        self._add_custom_model(cid, path)

    def _on_files_dropped(self, paths: list) -> None:
        unknown: list[str] = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                self._open_video(p)
            elif ext in MODEL_EXTENSIONS:
                cid = self._infer_class(p)
                if cid is not None:
                    self._add_custom_model(cid, p)
                else:
                    unknown.append(os.path.basename(p))
            elif os.path.isdir(p) and _detect_format(p) == "openvino":
                cid = self._infer_class(p)
                if cid is not None:
                    self._add_custom_model(cid, p)
                else:
                    unknown.append(os.path.basename(p))
            else:
                unknown.append(os.path.basename(p))
        if unknown:
            QtWidgets.QMessageBox.information(
                self, "Unrecognised files",
                "Could not auto-assign:\n" + "\n".join(unknown))

    @staticmethod
    def _infer_class(path: str) -> Optional[int]:
        n = os.path.splitext(os.path.basename(path))[0].lower()
        n = n.translate(_CLASS_INFER_TRANSLATE)
        for cid, keys in _CLASS_INFER_RULES:
            if any(k in n for k in keys):
                return cid
        return None

    # ── pipeline control ──────────────────────────────────────────────────

    def _open_video(self, path: str) -> None:
        self._on_stop()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Cannot open video:\n{path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read(); cap.release()
        if self._preview_cap is not None:
            self._preview_cap.release()
        self._preview_cap = cv2.VideoCapture(path)
        if ok and frame is not None:
            self._show_frame(frame)
            fh, fw = frame.shape[:2]
            self._fit_window_to_video_aspect(fw, fh)
        self._state.set_video_source(path)
        self._video_fps, self._total_frames = float(fps), total
        self._init_seek()
        self.lblTime.setText("0:00")
        bn = os.path.basename(path)
        self._set_source_label(bn, path)
        self._set_camera_combo_none()
        self._set_status(
            f"Loaded: {bn}  \u00b7  "
            f"{self._fmt(total / fps if fps else 0)} @ {fps:.1f} FPS")

    def _open_camera(self, camera: dict, label: str) -> None:
        self._on_stop()
        # Quick open / close *just* to verify the device is reachable — we do
        # NOT read frames here.  Reading and releasing the camera leaves the
        # Windows DirectShow driver in an inconsistent state and is the main
        # cause of the "live camera doesn't display anything" bug.
        cap = open_camera_capture(camera)
        if not cap.isOpened():
            cap.release()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Cannot open camera:\n{label}")
            self._set_camera_combo_none()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if self._preview_cap is not None:
            self._preview_cap.release()
        self._preview_cap = None

        # Show a placeholder until the pipeline produces the first frame.
        self._last_pixmap = QtGui.QPixmap()
        self.videoDisplay.setPixmap(QtGui.QPixmap())
        self.videoDisplay.setText(f"Starting {label}…")

        self._state.set_camera_source(camera)
        self._video_fps = float(fps)
        self._total_frames = 0
        self._init_seek()
        self.lblTime.setText("0:00")
        self._set_source_label(label, str(camera.get("target") or label))
        self._set_status(f"Loaded: {label}  ·  Live @ {fps:.1f} FPS")

        # Auto-start playback for live cameras — there is no "paused" state
        # that makes sense for a webcam.
        self._ensure_pipeline()
        self._state.pause_event.clear()
        self._is_playing = True
        self.btnPlay.setIcon(self._ico_pause)

    def _on_play_pause(self) -> None:
        if not self._has_active_source():
            self._set_status("Load a video or select a camera first")
            return
        if self._is_playing:
            self._state.pause_event.set(); self._is_playing = False
            self._state.flush_queues()   # drain buffered frames immediately
            self.btnPlay.setIcon(self._ico_play)
            self._set_status("Paused")
        else:
            self._ensure_pipeline()
            self._state.pause_event.clear(); self._is_playing = True
            self.btnPlay.setIcon(self._ico_pause)
            self._set_status("Playing\u2026")

    def _on_stop(self) -> None:
        if self._grabber or self._inferencer or self._tracker:
            self._state.stop_event.set(); self._state.pause_event.clear()
            for t in (self._grabber, self._inferencer, self._tracker):
                if t and t.isRunning():
                    t.wait(2000)
            self._grabber = self._inferencer = self._tracker = None
            self._state.stop_event.clear(); self._state.flush_queues()
        self._is_playing = False
        self.btnPlay.setIcon(self._ico_play)

    def _on_close_video(self) -> None:
        self._on_stop()
        if self._preview_cap is not None:
            self._preview_cap.release(); self._preview_cap = None
        self._seekPrev.hide()
        self._state.set_video_source(None)
        self._state.set_camera_source(None)
        self._video_fps = self._total_frames = 0
        self.seekSlider.setRange(0, 0); self.seekSlider.setEnabled(False)
        self.lblTime.setText("0:00"); self.lblTotal.setText("0:00")
        self._last_pixmap = QtGui.QPixmap()
        self.videoDisplay.setPixmap(QtGui.QPixmap())
        self.videoDisplay.setText(_DROP_PROMPT)
        self.lblVidName.setText(""); self.lblVidName.setToolTip("")
        self._set_video_ui_state(False)
        self._set_camera_combo_none()
        self._set_status("Ready")

    def _ensure_pipeline(self) -> None:
        if self._grabber and self._grabber.isRunning():
            return

        from pipeline.frame_grabber import FrameGrabberThread
        from pipeline.inference_engine import InferenceThread
        from pipeline.tracker_logic import TrackerLogicThread

        self._state.stop_event.clear(); self._state.pause_event.clear()
        self._state.flush_queues(); self._state.reset_tracker_flag = True
        self._grabber = FrameGrabberThread(self._state, parent=self)
        self._inferencer = InferenceThread(self._state, parent=self)
        self._tracker = TrackerLogicThread(self._state, parent=self)
        self._grabber.metaReady.connect(self._on_meta)
        self._grabber.positionChanged.connect(self._on_position)
        self._grabber.finished_signal.connect(self._on_video_ended)
        self._grabber.error.connect(
            lambda msg: self._log_error(f"Error: {msg}"))
        self._inferencer.fps_update.connect(self._on_fps)
        self._inferencer.status.connect(self._log)
        self._tracker.stats_ready.connect(self._on_stats)
        self._tracker.violation_detected.connect(self._on_violation_event)
        self._tracker.start()
        self._inferencer.start()
        self._grabber.start()

    # ── seek / skip ───────────────────────────────────────────────────────

    def _skip(self, seconds: float) -> None:
        path, _cam = self._state.get_source()
        if not path or self._video_fps <= 0 or self._total_frames <= 1:
            return
        t = max(0, min(self.seekSlider.value()
                       + int(seconds * self._video_fps),
                       self._total_frames - 1))
        self._state.request_seek(t)

    def _on_seek_released(self) -> None:
        path, _cam = self._state.get_source()
        if not path:
            self._user_seeking = False
            return
        self._user_seeking = False
        if hasattr(self, "_scrub_timer"):
            self._scrub_timer.stop()
        self._state.request_seek(self.seekSlider.value())

    def _on_seek_moved(self, v: int) -> None:
        if self._video_fps > 0:
            self.lblTime.setText(self._fmt(v / self._video_fps))
        # Live-scrub: while the user is dragging, request the latest seek
        # at most every ~120 ms so the preview frame keeps up with the
        # cursor without flooding the grabber thread.
        if self._user_seeking and self._has_active_source():
            if not hasattr(self, "_scrub_timer"):
                self._scrub_timer = QtCore.QTimer(self)
                self._scrub_timer.setSingleShot(True)
                self._scrub_timer.setInterval(120)
                self._scrub_timer.timeout.connect(self._on_scrub_tick)
            self._pending_scrub = int(v)
            if not self._scrub_timer.isActive():
                self._scrub_timer.start()

    def _on_scrub_tick(self) -> None:
        if self._user_seeking and hasattr(self, "_pending_scrub"):
            self._state.request_seek(int(self._pending_scrub))

    # ── seek preview ──────────────────────────────────────────────────────

    def _on_seek_hover(self, x: int) -> None:
        path, _cam = self._state.get_source()
        if not path or self._total_frames <= 0:
            self._seekPrev.hide(); return
        w = self.seekSlider.width()
        if w <= 0:
            return
        ratio = max(0.0, min(1.0, x / w))
        frame_idx = int(ratio * (self._total_frames - 1))
        sec = frame_idx / max(self._video_fps, 1)
        self._prevTime.setText(self._fmt(sec))
        self._seekPrev.adjustSize()
        pw = self._seekPrev.width()
        ph = self._seekPrev.height()
        sp = self.seekSlider.mapTo(self._vc, QtCore.QPoint(x, 0))
        px = max(4, min(self._vc.width() - pw - 4, sp.x() - pw // 2))
        py = max(4, self._overlay.y() - ph - 8)
        self._seekPrev.move(px, py)
        self._seekPrev.show(); self._seekPrev.raise_()

    # ── display ───────────────────────────────────────────────────────────

    def _poll_display(self) -> None:
        try:
            packet = self._state.display_queue.get_nowait()
        except queue.Empty:
            packet = None
        if packet is not None:
            self._show_frame(packet.annotated_frame)

    def _show_frame(self, bgr: np.ndarray) -> None:
        self._last_rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self._last_rgb_frame
        h, w, c = rgb.shape
        self._last_pixmap = QtGui.QPixmap.fromImage(
            QtGui.QImage(rgb.data, w, h, c * w,
                         QtGui.QImage.Format_RGB888))
        mode = self._video_scale_mode()
        self.videoDisplay.setPixmap(self._last_pixmap.scaled(
            self.videoDisplay.size(), mode,
            QtCore.Qt.FastTransformation))

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_meta(self, fps: float, total: int) -> None:
        if fps > 0:
            self._video_fps = float(fps)
        self._total_frames = max(0, int(total))
        self._init_seek()

    def _on_position(self, idx: int, ts: float) -> None:
        if self._user_seeking:
            return
        if self.seekSlider.isEnabled():
            self.seekSlider.blockSignals(True)
            self.seekSlider.setValue(idx)
            self.seekSlider.blockSignals(False)
        display_ts = ts if ts > 0 else (idx / self._video_fps if self._video_fps > 0 else 0.0)
        self.lblTime.setText(self._fmt(display_ts))

    def _on_fps(self, fps: float) -> None:
        self._current_fps = fps

    def _on_stats(self, s: dict) -> None:
        parts = [f"FPS {self._current_fps:.0f}"]
        for k, lbl in _STAT_ITEMS:
            if s.get(k):
                parts.append(f"{lbl} {s[k]}")
        self.statusLabel.setText("  \u00b7  ".join(parts))

    def _on_video_ended(self) -> None:
        self._is_playing = False
        self.btnPlay.setIcon(self._ico_play)
        self._set_status("End of video")

    def _on_conf(self, v: int) -> None:
        self.confLabel.setText(f"{v / 100:.2f}")
        self._state.set_conf(v / 100)

    def _on_iou(self, v: int) -> None:
        self.iouLabel.setText(f"{v / 100:.2f}")
        self._state.set_iou(v / 100)

    def _on_imgsz_changed(self) -> None:
        size = int(self.imgszCombo.currentData() or 640)
        self._state.set_imgsz(size)
        self._set_status(f"Inference size set to {size} (takes effect on next start)")

    def _on_device_changed(self) -> None:
        device = str(self.deviceCombo.currentData() or "auto")
        self._state.set_device(device)
        # Re-resolve auto model for the new device
        combo = self._mcombo[_UNIFIED_KEY]
        data = combo.currentData()
        if data and data.get("type") == "auto":
            self._resolve_auto_model(_UNIFIED_KEY)
        self._update_convert_buttons()
        self._update_model_format_labels()
        self._set_status(f"Device set to {device.upper()} (takes effect on next start)")

    def _on_fp16_toggled(self, enabled: bool) -> None:
        self._state.set_use_fp16(bool(enabled))
        self._set_status(
            "FP16 preference enabled (used for optimization + CUDA inference)"
            if enabled else
            "FP16 preference disabled (use FP32)"
        )

    def _on_overload_toggled(self, enabled: bool) -> None:
        # Enabled = enforce 2-rider limit (any extra rider = overload).
        # Disabled = effectively never overload.
        self._state.set_max_riders_per_motorcycle(2 if enabled else 999)
        self._set_status(
            "Overload highlighting on (>2 riders)"
            if enabled else "Overload highlighting off"
        )

    def _on_db_logging_toggled(self, enabled: bool) -> None:
        _sblog.set_enabled(bool(enabled))
        if enabled and not _sblog.has_credentials():
            self._set_status(
                "DB logging requested but Supabase credentials are missing")
        else:
            self._set_status(
                "DB logging (Supabase) enabled" if enabled
                else "DB logging (Supabase) disabled"
            )

    def _on_stride_changed(self, v: int) -> None:
        self.strideLabel.setText(str(v))
        self._state.set_inference_stride(v)
        self._set_status(f"Inference stride set to {v}x")

    # ── model optimisation ────────────────────────────────────────────────

    def _get_current_device_key(self) -> str:
        """Return 'cuda' or 'cpu' based on current combo box selection."""
        return self._get_effective_device()

    def _update_convert_buttons(self) -> None:
        """Show / hide the unified convert button and the Optimize-All button."""
        rt = _detect_runtimes()
        has_cuda = self._has_cuda()

        probe_cid = next(iter(TARGET_CLASS_IDS))
        group = self._model_groups.get(probe_cid)
        btn = self._mconv[_UNIFIED_KEY]

        if not group or not group.has_pt:
            btn.setVisible(False)
            self.btnOptimizeAll.setVisible(False)
            return

        gpu_done = (
            (not has_cuda)
            or (rt.best_gpu_format == "pt")
            or bool(group.get_variant(rt.best_gpu_format))
        )
        cpu_done = (
            (rt.best_cpu_format == "pt")
            or bool(group.get_variant(rt.best_cpu_format))
        )
        all_done = gpu_done and cpu_done
        no_runtime = (rt.best_gpu_format == "pt" and rt.best_cpu_format == "pt")

        if all_done:
            btn.setVisible(True)
            btn.setEnabled(False)
            btn.setIcon(_fa("check-circle", "#4CAF50", int(11 * _S)))
            btn.setToolTip("Fully optimised")
            self.btnOptimizeAll.setVisible(False)
        elif no_runtime:
            btn.setVisible(True)
            btn.setEnabled(False)
            btn.setIcon(_fa("exclamation-triangle", "#888", int(11 * _S)))
            btn.setToolTip("No optimisation runtime installed (tensorrt / openvino)")
            self.btnOptimizeAll.setVisible(False)
        else:
            missing = []
            if has_cuda and not gpu_done:
                missing.append("GPU")
            if not cpu_done:
                missing.append("CPU")
            btn.setVisible(True)
            btn.setEnabled(True)
            btn.setIcon(_fa("bolt", _TD, int(11 * _S)))
            btn.setToolTip(f"Optimise for {', '.join(missing)}")
            self.btnOptimizeAll.setVisible(True)

    def _build_convert_jobs(self, cids=None):
        """Return ConvertJob tuples for the unified detector (GPU + CPU)."""
        probe_cid = next(iter(TARGET_CLASS_IDS))
        group = self._model_groups.get(probe_cid)
        if not group or not group.has_pt:
            return []
        pt_variant = group.get_variant("pt")
        if not pt_variant:
            return []

        imgsz = self._state.get_imgsz()
        has_cuda = self._has_cuda()
        half = self._state.use_fp16()
        jobs = []
        if has_cuda:
            jobs.append((_UNIFIED_KEY, pt_variant.path, "cuda", imgsz, half))
        jobs.append((_UNIFIED_KEY, pt_variant.path, "cpu", imgsz, half))
        return jobs

    def _on_convert_model(self, cid: int = _UNIFIED_KEY) -> None:
        """Start conversion of the unified detector (GPU + CPU)."""
        jobs = self._build_convert_jobs()
        if not jobs:
            self._set_status("Nothing to convert (already optimised or no .pt available)")
            return
        rt = _detect_runtimes()
        parts = []
        if self._has_cuda():
            parts.append(rt.gpu_summary)
        parts.append(rt.cpu_summary)
        self._set_status("Starting conversion · " + " | ".join(parts))
        self._start_conversion(jobs)

    def _on_convert_all(self) -> None:
        """Alias for _on_convert_model under the unified-detector layout."""
        self._on_convert_model(_UNIFIED_KEY)

    def _start_conversion(self, jobs) -> None:
        if self._converter and self._converter.isRunning():
            self._set_status("A conversion is already running — please wait")
            return

        from pipeline.convert_worker import ConvertWorker

        self._converter = ConvertWorker(jobs)   # no parent — we manage lifetime
        self._converter.conversion_started.connect(self._on_cvt_started)
        self._converter.conversion_progress.connect(self._on_cvt_progress)
        self._converter.conversion_finished.connect(self._on_cvt_finished)
        self._converter.all_finished.connect(self._on_cvt_all_done)
        # Disable convert buttons while running
        self._mconv[_UNIFIED_KEY].setEnabled(False)
        self.btnOptimizeAll.setEnabled(False)
        self._converter.start()

    def _on_cvt_started(self, cid: int, msg: str) -> None:
        import time
        self._cvt_t0 = time.monotonic()
        self._cvt_label = msg
        self._set_status(msg)
        self._cvt_timer.start()   # kick off heartbeat
        btn = self._mconv.get(cid)
        if btn:
            btn.setIcon(_fa("spinner", "#FFC107", int(11 * _S)))
            btn.setToolTip("Converting…")

    def _on_cvt_heartbeat(self) -> None:
        """Called every 20 s while a conversion is in progress."""
        import time
        if not (self._converter and self._converter.isRunning()):
            self._cvt_timer.stop()
            return
        elapsed = int(time.monotonic() - self._cvt_t0)
        self._set_status(
            f"{self._cvt_label}  ({elapsed}s elapsed — TRT builds take 3-5 min)")

    def _on_cvt_progress(self, cid: int, msg: str) -> None:
        self._cvt_label = msg   # keep heartbeat message in sync
        self._set_status(msg)

    def _on_cvt_finished(self, cid: int, success: bool, result: str) -> None:
        # Infer format from the result path (works for both GPU and CPU jobs)
        r = result.lower()
        if ".engine" in r:
            fmt_name = "TensorRT"
        elif "openvino" in r:
            fmt_name = "OpenVINO"
        elif r.endswith(".onnx"):
            fmt_name = "ONNX"
        else:
            fmt_name = "optimised"

        if success:
            self._set_status(
                f"{CLASS_NAMES.get(cid, 'Unified Detector')} → {fmt_name} ✓  ({os.path.basename(result)})")
            btn = self._mconv.get(cid)
            if btn:
                btn.setIcon(_fa("check-circle", "#4CAF50", int(11 * _S)))
                btn.setToolTip(f"Optimised ({fmt_name})")
                btn.setEnabled(False)
        else:
            self._set_status(
                f"{CLASS_NAMES.get(cid, 'Unified Detector')} conversion failed: {result.splitlines()[0]}")
            btn = self._mconv.get(cid)
            if btn:
                btn.setIcon(_fa("exclamation-triangle", "#f44336", int(11 * _S)))
                btn.setToolTip(f"Conversion failed — click to retry")
                btn.setEnabled(True)

    def _on_cvt_all_done(self) -> None:
        """All jobs finished — refresh buttons and trigger model hot-reload."""
        self._cvt_timer.stop()
        # Re-discover models (new optimised variants now exist)
        self._refresh_model_combos()
        self._update_convert_buttons()
        self._mconv[_UNIFIED_KEY].setEnabled(True)
        self.btnOptimizeAll.setEnabled(True)
        # Tell inference engine to reload with the new optimised weights
        self._state.reload_models_flag = True
        self._set_status("Model optimisation complete — inference will reload")

    def _show_speed_menu(self) -> None:
        pass  # replaced by spdSlider – kept so old references don't crash

    def _on_speed_changed(self, idx: int) -> None:
        self._spd_idx = idx
        rate = self._speeds[idx]
        label = f"{rate:g}×"
        self.spdLabel.setText(label)
        self._state.set_playback_rate(rate)

    # ── helpers ───────────────────────────────────────────────────────────

    def _init_seek(self) -> None:
        if self._total_frames > 1:
            self.seekSlider.setEnabled(True)
            self.seekSlider.setRange(0, self._total_frames - 1)
        else:
            self.seekSlider.setEnabled(False)
            self.seekSlider.setRange(0, 0)
            self.lblTotal.setText("0:00")
        if self._video_fps > 0 and self._total_frames > 0:
            self.lblTotal.setText(
                self._fmt((self._total_frames - 1) / self._video_fps))

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int((sec or 0) + 0.5))
        return f"{sec // 60}:{sec % 60:02d}"

    def _set_status(self, t: str) -> None:
        # Bottom bar is now reserved for live stats (FPS, motorcycle/rider
        # counts, etc.). Free-form status messages go to the Console panel.
        self._log(t)

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.isLocalFile()]
        if paths:
            self._on_files_dropped(paths)
            e.acceptProposedAction()

    def closeEvent(self, e):
        self._on_stop()
        try:
            _sblog.shutdown(timeout=1.0)
        except Exception:
            pass
        if self._cam_scan and self._cam_scan.isRunning():
            self._cam_scan.wait(2000)
        # Stop conversion worker before closing (avoids crash from live QThread)
        if self._converter and self._converter.isRunning():
            self._cvt_timer.stop()
            self._converter.request_stop()      # kill child subprocess + set flag
            # Disconnect signals first so no slots fire on dead objects
            try:
                self._converter.conversion_started.disconnect()
                self._converter.conversion_progress.disconnect()
                self._converter.conversion_finished.disconnect()
                self._converter.all_finished.disconnect()
            except Exception:
                pass
            self._converter.wait(5000)          # give it up to 5 s gracefully
            if self._converter.isRunning():
                self._converter.terminate()     # force-kill if still running
                self._converter.wait(1000)
        if self._preview_cap is not None:
            self._preview_cap.release(); self._preview_cap = None
        e.accept()

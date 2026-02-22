"""
Main GUI – Miruro-style monotone dark media player + YOLO pipeline.
Controls overlay the video. Fixed 1.5× base scale (no dynamic rescaling).
True fullscreen with auto-hiding transport controls.
"""
from __future__ import annotations
import os, queue, tempfile
from typing import Dict, Optional
import cv2, numpy as np, qtawesome as qta
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from config import (TARGET_CLASS_IDS, CLASS_NAMES, VIDEO_EXTENSIONS,
                    MODEL_EXTENSIONS, DEFAULT_MODEL_FILES,
                    OPTIMIZED_GPU_DIR, OPTIMIZED_CPU_DIR)
from pipeline.state import PipelineState
from pipeline.frame_grabber import FrameGrabberThread
from pipeline.inference_engine import InferenceThread
from pipeline.tracker_logic import TrackerLogicThread
from pipeline.convert_worker import ConvertWorker
from utils.runtime_check import detect as _detect_runtimes
from gui.widgets import VideoDropLabel, SeekSlider

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Palette ──────────────────────────────────────────────────────────────────
_BG = "#0d0d0d"; _BG_SB = "#0f0f0f"; _BD = "#1e1e1e"
_T = "#b0b0b0"; _TD = "#555555"; _TH = "#e0e0e0"; _W = "#ffffff"
_HOV = "rgba(255,255,255,0.07)"; _PRS = "rgba(255,255,255,0.14)"

def _fa(name, color=_T, sz=18):
    return qta.icon(f"fa5s.{name}", color=color)

_S = 1.5  # fixed UI scale – no dynamic rescaling
_DROP_PROMPT = "Drop a video here or use the sidebar to load one"
_FS_ICON_SIZE = int(16 * _S)
_STAT_ITEMS = (
    ("motorcycles", "Motos"),
    ("riders", "Riders"),
    ("no_helmet", "No Helmet"),
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


def _qss(s: float, chk: str) -> str:
    """Full stylesheet – every pixel value multiplied by *s*."""
    def px(v): return f"{max(1, round(v * s))}px"
    return (
        f"QMainWindow,QWidget{{background:{_BG};color:{_T};"
        f"font-family:'Segoe UI','Arial',sans-serif;font-size:{px(12)};}}"
        f"#topBar{{background:{_BG};border-bottom:1px solid {_BD};}}"
        f"#topBar QLabel{{background:transparent;}}"
        f"#topBarTitle{{color:{_TH};font-size:{px(13)};font-weight:600;"
        f"background:transparent;}}"
        f"#sidebar{{background:{_BG_SB};border-right:1px solid {_BD};}}"
        f"#sidebar QLabel{{color:{_T};}}"
        f"#sidebarTitle{{font-size:{px(13)};font-weight:600;color:{_TH};"
        f"padding:{px(6)} 0 {px(10)} 0;letter-spacing:0.5px;}}"
        f"#sectionLabel{{font-size:{px(10)};font-weight:600;color:{_TD};"
        f"text-transform:uppercase;letter-spacing:1.2px;"
        f"padding:{px(14)} 0 {px(4)} 0;}}"
        f"#modelCard{{background:transparent;border:none;"
        f"border-radius:0;padding:{px(2)} 0;margin:0;}}"
        f"#modelFileBox{{background:transparent;border-radius:0;"
        f"border-left:2px solid #333;}}"
        f"#modelFileBox *{{background:transparent;}}"
        f"#modelFileIcon{{background:transparent;border:none;min-width:0;"
        f"padding:0;margin:0;}}"
        f"#modelFileIcon:hover{{background:rgba(255,255,255,0.06);"
        f"border-radius:{px(2)};}}"
        f"#modelFile{{color:{_TD};font-size:{px(11)};"
        f"padding:{px(2)} {px(6)};background:transparent;}}"
        f"#modelFileLoaded{{color:{_TH};font-size:{px(11)};"
        f"padding:{px(2)} {px(6)};background:transparent;}}"
        f"#videoFileRow{{background:transparent;border:none;"
        f"border-radius:0;}}"
        f"#videoFileRow *{{background:transparent;}}"
        f"#videoFileRow:hover{{background:rgba(255,255,255,0.04);}}"
        f"QPushButton{{background:{_BG};color:{_T};border:1px solid {_BD};"
        f"border-radius:{px(4)};padding:{px(5)} {px(12)};"
        f"font-size:{px(11)};}}"
        f"QPushButton:hover{{background:#1a1a1a;}}"
        f"QPushButton:pressed{{background:#222;}}"
        f"QPushButton:disabled{{background:{_BG};color:#333;border-color:#151515;}}"
        f"QPushButton:focus{{outline:none;}}"
        f"#iconBtn{{background:transparent;border:none;"
        f"border-radius:{px(4)};padding:0;"
        f"outline:none;}}"
        f"#iconBtn:hover{{background:{_HOV};}}"
        f"#iconBtn:pressed{{background:{_PRS};}}"
        f"#iconBtn:focus{{outline:none;background:transparent;}}"
        f"#closeVideoBtn{{background:transparent;border:none;"
        f"padding:0;margin:0;min-width:0;}}"
        f"#closeVideoBtn:hover{{background:rgba(255,255,255,0.08);"
        f"border-radius:{px(3)};}}"
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
        # sidebar sliders
        f"QSlider::groove:horizontal{{height:{px(3)};background:#2a2a2a;"
        f"border-radius:{px(1)};}}"
        f"QSlider::handle:horizontal{{background:{_TH};border:none;"
        f"width:{px(10)};height:{px(10)};margin:-{px(4)} 0;"
        f"border-radius:{px(5)};}}"
        f"QSlider::handle:horizontal:hover{{background:{_W};}}"
        f"QSlider::sub-page:horizontal{{background:{_T};"
        f"border-radius:{px(1)};}}"
        f"QCheckBox{{spacing:{px(6)};color:{_T};font-size:{px(11)};}}"
        f"QCheckBox::indicator{{width:{px(13)};height:{px(13)};"
        f"border:1px solid #444;border-radius:{px(3)};background:#1a1a1a;}}"
        f"QCheckBox::indicator:checked{{background:#1a1a1a;"
        f"border:1px solid #555;border-radius:{px(3)};"
        f"image:url({chk});}}"
        f"#videoDisplay{{background:#000;border:none;color:{_TD};"
        f"font-size:{px(13)};}}"
        f"#overlayCtrl,#overlaySeek,#ctrlBar{{background:transparent;}}"
        f"#overlayCtrl QWidget{{background:transparent;}}"
        f"#timeLabel{{color:rgba(255,255,255,0.6);font-size:{px(11)};"
        f"font-family:'Consolas','Courier New',monospace;"
        f"padding:0;margin:0;background:transparent;}}"

        f"#vidName{{color:{_T};font-size:{px(11)};padding:0;"
        f"background:transparent;}}"
        f"#sliderLabel{{color:{_TD};font-size:{px(11)};}}"
        f"QComboBox{{background:transparent;color:{_T};"
        f"border:1px solid #1a1a1a;border-radius:{px(3)};"
        f"padding:{px(4)} {px(8)};font-size:{px(11)};"
        f"min-height:{px(20)};}}"
        f"QComboBox:hover{{border-color:#333;}}"
        f"QComboBox::drop-down{{border:none;width:{px(20)};}}"
        f"QComboBox::down-arrow{{image:none;}}"
        f"QComboBox QAbstractItemView{{background:#141414;color:{_T};"
        f"border:1px solid {_BD};selection-background-color:#222;"
        f"font-size:{px(11)};}}"
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
    )


class MainWindow(QtWidgets.QMainWindow):
    """Miruro-style monotone dark media player + YOLO pipeline controller."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Safety Gear Compliance")
        self.setMinimumSize(1440, 750)
        self.resize(1440, 750)
        self.setAcceptDrops(True)
        self._chk = self._make_checkmark_icon()
        self._state = PipelineState()
        self._grabber: Optional[FrameGrabberThread] = None
        self._inferencer: Optional[InferenceThread] = None
        self._tracker: Optional[TrackerLogicThread] = None
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
        # model conversion worker
        self._converter: Optional[ConvertWorker] = None
        self._cvt_t0: float = 0.0          # monotonic start time of conversion
        self._cvt_label: str = ""           # last "Converting X → Y" message
        # heartbeat timer: update status bar every 20 s during long TRT builds
        self._cvt_timer = QtCore.QTimer(self)
        self._cvt_timer.setInterval(20_000)
        self._cvt_timer.timeout.connect(self._on_cvt_heartbeat)
        self._fs_hide.setSingleShot(True)
        self._fs_hide.setInterval(3000)
        self._fs_hide.timeout.connect(self._fs_hide_overlay)
        self._build_ui()
        self._setup_shortcuts()
        self.setStyleSheet(_qss(_S, self._chk))
        self._auto_load_models()
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

    def _set_model_color_btn_style(self, cid: int) -> None:
        """Apply class colour accent and status text colour for model entry."""
        lbl = self._mstat[cid]
        box = self._mfile_box[cid]
        qcol = self._to_qcolor_bgr(self._state.get_class_color(cid))
        loaded = lbl.objectName() == "modelFileLoaded"
        fg = _TH if loaded else _TD
        box.setStyleSheet(
            f"QWidget#modelFileBox{{background:transparent;border-radius:0;"
            f"border-left:2px solid {qcol.name()};}}")
        lbl.setStyleSheet(
            f"QLabel{{color:{fg};font-size:{int(11*_S)}px;"
            f"padding:{int(2*_S)}px {int(6*_S)}px;"
            "background:transparent;}")

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
        self._splitter.addWidget(self.sidebar)

        main = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(main)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)
        self._splitter.addWidget(main)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setCollapsible(0, False)
        self._splitter.setSizes([int(230 * _S), int(730 * _S)])

        # top bar
        self._topBar = tb = QtWidgets.QWidget()
        tb.setObjectName("topBar"); tb.setFixedHeight(int(38 * _S))
        tr = QtWidgets.QHBoxLayout(tb)
        tr.setContentsMargins(int(12*_S), 0, int(10*_S), 0)
        tr.setSpacing(int(8 * _S))
        title = QtWidgets.QLabel("Safety Gear Compliance")
        title.setObjectName("topBarTitle")
        tr.addWidget(title); tr.addStretch()
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

        # speed
        self._speeds = [("0.25×", .25), ("0.5×", .5), ("1×", 1.),
                        ("1.5×", 1.5), ("2×", 2.), ("4×", 4.)]
        self._spd_idx = 2
        self.btnSpeed = self._ibtn("tachometer-alt", "Playback speed")

        for w in (self.btnPlay, self.btnStepFwd, self._btnVol, self.btnSpeed):
            cr.addWidget(w, 0, AV)

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
        sb.setMinimumWidth(int(200 * _S))
        sb.setMaximumWidth(int(340 * _S))
        outer = QtWidgets.QVBoxLayout(sb)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(
            int(16*_S), int(14*_S), int(16*_S), int(14*_S))
        lay.setSpacing(int(4 * _S))

        hdr = QtWidgets.QLabel("Controls")
        hdr.setObjectName("sidebarTitle"); lay.addWidget(hdr)

        # VIDEO
        lay.addWidget(self._section("VIDEO"))
        self.btnLoadVideo = QtWidgets.QPushButton()
        isz = int(13 * _S)
        self.btnLoadVideo.setIcon(_fa("folder-open", _T, isz))
        self.btnLoadVideo.setIconSize(QtCore.QSize(isz, isz))
        self.btnLoadVideo.setText("  Load Video")
        self.btnLoadVideo.setFixedHeight(int(28 * _S))
        self.btnLoadVideo.setFocusPolicy(QtCore.Qt.NoFocus)
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
        lay.addSpacing(int(6 * _S))

        # MODELS
        lay.addWidget(self._section("MODELS"))
        self._mbtn: Dict[int, QtWidgets.QPushButton] = {}
        self._mstat: Dict[int, QtWidgets.QLabel] = {}
        self._mtog: Dict[int, QtWidgets.QCheckBox] = {}
        self._mfile_box: Dict[int, QtWidgets.QWidget] = {}
        self._mconv: Dict[int, QtWidgets.QPushButton] = {}  # convert buttons
        mcard = QtWidgets.QWidget(); mcard.setObjectName("modelCard")
        ml = QtWidgets.QVBoxLayout(mcard)
        ml.setContentsMargins(0, int(4*_S), 0, int(4*_S))
        ml.setSpacing(int(5 * _S))
        bisz = int(11 * _S)
        for cid in TARGET_CLASS_IDS:
            name = CLASS_NAMES[cid]
            block = QtWidgets.QWidget()
            tv = QtWidgets.QVBoxLayout(block)
            tv.setContentsMargins(0, 0, 0, 0)
            tv.setSpacing(int(3 * _S))
            chk = QtWidgets.QCheckBox(name)
            chk.setChecked(True)
            self._mtog[cid] = chk
            tv.addWidget(chk)
            file_box = QtWidgets.QWidget(); file_box.setObjectName("modelFileBox")
            self._mfile_box[cid] = file_box
            fr = QtWidgets.QHBoxLayout(file_box)
            fr.setContentsMargins(0, 0, 0, 0)
            fr.setSpacing(0)
            st = QtWidgets.QLabel("\u2014")
            st.setObjectName("modelFile")
            st.setToolTip("Click to change colour")
            st.setWordWrap(False)
            st.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            st.setMinimumHeight(int(22 * _S))
            st.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._mstat[cid] = st
            fr.addWidget(st, stretch=1)

            _msq = int(22 * _S)  # square 1:1
            btn = QtWidgets.QPushButton()
            btn.setObjectName("modelFileIcon")
            btn.setIcon(_fa("folder-open", _TD, bisz))
            btn.setIconSize(QtCore.QSize(bisz, bisz))
            btn.setFixedSize(_msq, _msq)
            btn.setToolTip(f"Load {name} model")
            btn.setFlat(True)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._mbtn[cid] = btn
            fr.addWidget(btn)

            # convert / optimise button (visible only for .pt models)
            conv = QtWidgets.QPushButton()
            conv.setObjectName("modelFileIcon")
            conv.setIcon(_fa("bolt", _TD, bisz))
            conv.setIconSize(QtCore.QSize(bisz, bisz))
            conv.setFixedSize(_msq, _msq)
            conv.setToolTip(f"Optimise {name} model for current device")
            conv.setFlat(True)
            conv.setFocusPolicy(QtCore.Qt.NoFocus)
            conv.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            conv.setVisible(False)
            self._mconv[cid] = conv
            fr.addWidget(conv)

            self._set_model_color_btn_style(cid)
            tv.addWidget(file_box)
            ml.addWidget(block)
        lay.addWidget(mcard)

        # "Optimize All" button – converts every loaded .pt model
        self.btnOptimizeAll = QtWidgets.QPushButton()
        _oisz = int(13 * _S)
        self.btnOptimizeAll.setIcon(_fa("bolt", _T, _oisz))
        self.btnOptimizeAll.setIconSize(QtCore.QSize(_oisz, _oisz))
        self.btnOptimizeAll.setText("  Optimize All Models")
        self.btnOptimizeAll.setFixedHeight(int(28 * _S))
        self.btnOptimizeAll.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnOptimizeAll.setVisible(False)
        lay.addWidget(self.btnOptimizeAll)

        lay.addSpacing(int(6 * _S))

        # DETECTION
        lay.addWidget(self._section("DETECTION"))
        dcard = QtWidgets.QWidget(); dcard.setObjectName("modelCard")
        dl = QtWidgets.QVBoxLayout(dcard)
        dl.setContentsMargins(0, int(4*_S), 0, int(4*_S))
        dl.setSpacing(int(5 * _S))

        self.chkOverlay = QtWidgets.QCheckBox("Show overlays")
        self.chkOverlay.setChecked(True)
        dl.addWidget(self.chkOverlay)

        for attr, label, default in [("conf", "Confidence", 25),
                                     ("iou", "IoU", 45)]:
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
            val.setFixedWidth(int(32 * _S))
            r.addWidget(slider, stretch=1); r.addWidget(val)
            dl.addLayout(r)
            setattr(self, f"{attr}Slider", slider)
            setattr(self, f"{attr}Label", val)

        perf_lbl = QtWidgets.QLabel("Inference Size")
        perf_lbl.setObjectName("sliderLabel")
        dl.addWidget(perf_lbl)
        self.imgszCombo = QtWidgets.QComboBox()
        for size in (256, 320, 480, 640, 960, 1280):
            self.imgszCombo.addItem(str(size), size)
        self.imgszCombo.setCurrentText("256")
        dl.addWidget(self.imgszCombo)

        dev_lbl = QtWidgets.QLabel("Device")
        dev_lbl.setObjectName("sliderLabel")
        dl.addWidget(dev_lbl)
        self.deviceCombo = QtWidgets.QComboBox()
        self.deviceCombo.addItem("Auto", "auto")
        self.deviceCombo.addItem("GPU (CUDA)", "cuda")
        self.deviceCombo.addItem("CPU", "cpu")
        dl.addWidget(self.deviceCombo)

        self.chkFp16 = QtWidgets.QCheckBox("Use FP16 (GPU only)")
        self.chkFp16.setChecked(False)
        dl.addWidget(self.chkFp16)

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
        self.strideLabel.setFixedWidth(int(20 * _S))
        sr.addWidget(self.strideSlider, stretch=1)
        sr.addWidget(self.strideLabel)
        dl.addLayout(sr)

        lay.addWidget(dcard)
        lay.addStretch()
        scroll.setWidget(content); outer.addWidget(scroll)
        return sb

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
                self.videoDisplay.setPixmap(self._last_pixmap.scaled(
                    self.videoDisplay.size(), QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.FastTransformation))
        # fullscreen: mouse movement on video/overlay → show controls
        if self._is_fs and etype == QtCore.QEvent.MouseMove:
            if obj in (self._vc, self.videoDisplay, self._overlay):
                self._fs_show_controls()
        # seek slider hover for preview
        if obj is self.seekSlider:
            if etype == QtCore.QEvent.MouseMove:
                self._on_seek_hover(event.pos().x())
            elif etype == QtCore.QEvent.Leave:
                self._seekPrev.hide()

        if obj is self.videoDisplay and etype == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton and self._state.video_path:
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

    # ── fullscreen ────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self._is_fs:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        self._is_fs = True
        self.sidebar.hide()
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
        self.sidebar.show()
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
        for cid, btn in self._mbtn.items():
            btn.clicked.connect(
                lambda _=False, c=cid: self._on_load_model(c))
        for cid, conv in self._mconv.items():
            conv.clicked.connect(
                lambda _=False, c=cid: self._on_convert_model(c))
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
        self.btnSpeed.clicked.connect(self._show_speed_menu)
        self.chkOverlay.toggled.connect(self._state.set_overlay_enabled)
        for cid, chk in self._mtog.items():
            chk.toggled.connect(
                lambda v, c=cid: self._state.set_model_enabled(c, v))
        for cid, lbl in self._mstat.items():
            lbl.mousePressEvent = (
                lambda ev, c=cid: self._on_pick_model_color(c)
                if ev.button() == QtCore.Qt.LeftButton else None)
        self.confSlider.valueChanged.connect(self._on_conf)
        self.iouSlider.valueChanged.connect(self._on_iou)
        self.imgszCombo.currentIndexChanged.connect(self._on_imgsz_changed)
        self.deviceCombo.currentIndexChanged.connect(self._on_device_changed)
        self.chkFp16.toggled.connect(self._on_fp16_toggled)
        self.strideSlider.valueChanged.connect(self._on_stride_changed)

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
        self._set_model_color_btn_style(cid)

    def _step_forward(self) -> None:
        if self._state.video_path:
            self._skip(1 / max(self._video_fps, 1))

    # ── auto-load models ──────────────────────────────────────────────────

    def _auto_load_models(self) -> None:
        d = os.path.join(_ROOT, "models"); loaded = []
        for cid, fn in DEFAULT_MODEL_FILES.items():
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                self._set_model(cid, p)
                loaded.append(CLASS_NAMES[cid])
        use_cuda = torch.cuda.is_available()
        default_imgsz = 320 if use_cuda else 256
        default_stride = 3 if use_cuda else 6
        self._state.set_imgsz(default_imgsz)
        self._state.set_device("cuda" if use_cuda else "auto")
        self._state.set_use_fp16(use_cuda)
        self._state.set_inference_stride(default_stride)

        self.imgszCombo.setCurrentText(str(default_imgsz))
        self.strideSlider.setValue(default_stride)
        self.strideLabel.setText(str(default_stride))
        self.deviceCombo.setCurrentIndex(1 if use_cuda else 0)
        self.chkFp16.setChecked(use_cuda)
        self._set_status(
            f"Auto-loaded: {', '.join(loaded)}" if loaded
            else "No default models found \u2014 load via sidebar")
        self._update_convert_buttons()

    # ── video / model loading ─────────────────────────────────────────────

    def _on_load_video(self) -> None:
        source_dir = os.path.join(_ROOT, "source")
        start_dir = source_dir if os.path.isdir(source_dir) else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All (*)")
        if p:
            self._open_video(p)

    def _on_load_model(self, cid: int) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load {CLASS_NAMES.get(cid, str(cid))} Model", "",
            "Model Files (*.pt *.engine *.onnx);;All (*)")
        if p:
            self._set_model(cid, p)

    def _set_model(self, cid: int, path: str) -> None:
        self._state.model_paths[cid] = path
        st = self._mstat[cid]
        base = os.path.basename(path)
        fm = st.fontMetrics()
        text_w = max(20, (st.width() if st.width() > 0 else int(120 * _S))
                     - int(12 * _S))
        st.setText(fm.elidedText(base, QtCore.Qt.ElideMiddle, text_w))
        st.setObjectName("modelFileLoaded")
        st.style().unpolish(st); st.style().polish(st)
        st.setToolTip(path)
        self._set_model_color_btn_style(cid)
        self._update_convert_buttons()

    def _on_files_dropped(self, paths: list) -> None:
        unknown: list[str] = []
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                self._open_video(p)
            elif ext in MODEL_EXTENSIONS:
                cid = self._infer_class(p)
                if cid is not None:
                    self._set_model(cid, p)
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
        self._state.video_path = path
        self._video_fps, self._total_frames = float(fps), total
        self._init_seek()
        bn = os.path.basename(path)
        self.lblVidName.setText(bn); self.lblVidName.setToolTip(path)
        self._set_video_ui_state(True)
        self._set_status(
            f"Loaded: {bn}  \u00b7  "
            f"{self._fmt(total / fps if fps else 0)} @ {fps:.1f} FPS")

    def _on_play_pause(self) -> None:
        if not self._state.video_path:
            self._set_status("Load a video first"); return
        if self._is_playing:
            self._state.pause_event.set(); self._is_playing = False
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
        self._state.video_path = None
        self._video_fps = self._total_frames = 0
        self.seekSlider.setRange(0, 0); self.seekSlider.setEnabled(False)
        self.lblTime.setText("0:00"); self.lblTotal.setText("0:00")
        self._last_pixmap = QtGui.QPixmap()
        self.videoDisplay.setPixmap(QtGui.QPixmap())
        self.videoDisplay.setText(_DROP_PROMPT)
        self.lblVidName.setText(""); self.lblVidName.setToolTip("")
        self._set_video_ui_state(False)
        self._set_status("Ready")

    def _ensure_pipeline(self) -> None:
        if self._grabber and self._grabber.isRunning():
            return
        self._state.stop_event.clear(); self._state.pause_event.clear()
        self._state.flush_queues(); self._state.reset_tracker_flag = True
        self._grabber = FrameGrabberThread(self._state, parent=self)
        self._inferencer = InferenceThread(self._state, parent=self)
        self._tracker = TrackerLogicThread(self._state, parent=self)
        self._grabber.metaReady.connect(self._on_meta)
        self._grabber.positionChanged.connect(self._on_position)
        self._grabber.finished_signal.connect(self._on_video_ended)
        self._grabber.error.connect(
            lambda msg: self._set_status(f"Error: {msg}"))
        self._inferencer.fps_update.connect(self._on_fps)
        self._inferencer.status.connect(self._set_status)
        self._tracker.stats_ready.connect(self._on_stats)
        self._tracker.start()
        self._inferencer.start()
        self._grabber.start()

    # ── seek / skip ───────────────────────────────────────────────────────

    def _skip(self, seconds: float) -> None:
        if self._video_fps <= 0:
            return
        t = max(0, min(self.seekSlider.value()
                       + int(seconds * self._video_fps),
                       self._total_frames - 1))
        self._state.request_seek(t)

    def _on_seek_released(self) -> None:
        self._user_seeking = False
        self._state.request_seek(self.seekSlider.value())

    def _on_seek_moved(self, v: int) -> None:
        if self._video_fps > 0:
            self.lblTime.setText(self._fmt(v / self._video_fps))

    # ── seek preview ──────────────────────────────────────────────────────

    def _on_seek_hover(self, x: int) -> None:
        if not self._state.video_path or self._total_frames <= 0:
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
        packet = None
        while True:
            try:
                packet = self._state.display_queue.get_nowait()
            except queue.Empty:
                break
        if packet is not None:
            self._show_frame(packet.annotated_frame)

    def _show_frame(self, bgr: np.ndarray) -> None:
        self._last_rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self._last_rgb_frame
        h, w, c = rgb.shape
        self._last_pixmap = QtGui.QPixmap.fromImage(
            QtGui.QImage(rgb.data, w, h, c * w,
                         QtGui.QImage.Format_RGB888))
        self.videoDisplay.setPixmap(self._last_pixmap.scaled(
            self.videoDisplay.size(), QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.FastTransformation))

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_meta(self, fps: float, total: int) -> None:
        if fps > 0:
            self._video_fps = float(fps)
        if total > 0:
            self._total_frames = int(total)
        self._init_seek()

    def _on_position(self, idx: int, ts: float) -> None:
        if self._user_seeking:
            return
        self.seekSlider.blockSignals(True)
        self.seekSlider.setValue(idx)
        self.seekSlider.blockSignals(False)
        self.lblTime.setText(self._fmt(ts))

    def _on_fps(self, fps: float) -> None:
        self._current_fps = fps

    def _on_stats(self, s: dict) -> None:
        parts = [f"FPS {self._current_fps:.0f}"]
        for k, lbl in _STAT_ITEMS:
            if s.get(k):
                parts.append(f"{lbl} {s[k]}")
        self._set_status("  \u00b7  ".join(parts))

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
        self._update_convert_buttons()
        self._set_status(f"Device set to {device.upper()} (takes effect on next start)")

    def _on_fp16_toggled(self, enabled: bool) -> None:
        self._state.set_use_fp16(bool(enabled))
        self._set_status("FP16 enabled (GPU only, takes effect on next start)"
                         if enabled else "FP16 disabled")

    def _on_stride_changed(self, v: int) -> None:
        self.strideLabel.setText(str(v))
        self._state.set_inference_stride(v)
        self._set_status(f"Inference stride set to {v}x")

    # ── model optimisation ────────────────────────────────────────────────

    def _get_current_device_key(self) -> str:
        """Return 'cuda' or 'cpu' based on current combo box selection."""
        raw = str(self.deviceCombo.currentData() or "auto").lower()
        if raw == "cpu":
            return "cpu"
        if raw in ("cuda", "auto") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _optimized_exists_for(self, cid: int, dev: str) -> bool:
        """Check whether an optimised model exists for *cid* + specific *dev*."""
        path = self._state.model_paths.get(cid)
        if not path or not path.lower().endswith(".pt"):
            return False
        basename = os.path.splitext(os.path.basename(path))[0]
        if dev == "cuda":
            return (
                os.path.isfile(os.path.join(OPTIMIZED_GPU_DIR, f"{basename}.engine"))
                or os.path.isfile(os.path.join(OPTIMIZED_GPU_DIR, f"{basename}.onnx"))
            )
        else:
            return (
                os.path.isdir(os.path.join(OPTIMIZED_CPU_DIR, f"{basename}_openvino_model"))
                or os.path.isfile(os.path.join(OPTIMIZED_CPU_DIR, f"{basename}.onnx"))
            )

    def _optimized_exists(self, cid: int) -> bool:
        """Check whether an optimised model exists for *cid* + current device."""
        return self._optimized_exists_for(cid, self._get_current_device_key())

    def _update_convert_buttons(self) -> None:
        """Show / hide per-model convert buttons and the Optimize-All button."""
        rt = _detect_runtimes()
        has_cuda = torch.cuda.is_available()

        # Build a combined tooltip string for what will be converted
        parts = []
        if has_cuda and rt.best_gpu_format != "pt":
            ver = rt.tensorrt_ver if rt.best_gpu_format == "engine" else rt.onnx_ver
            tag = {"engine": "TensorRT", "onnx": "ONNX"}.get(rt.best_gpu_format, rt.best_gpu_format.upper())
            parts.append(f"GPU → {tag} {ver}".strip())
        if rt.best_cpu_format != "pt":
            ver = rt.openvino_ver if rt.best_cpu_format == "openvino" else rt.onnx_ver
            tag = {"openvino": "OpenVINO", "onnx": "ONNX"}.get(rt.best_cpu_format, rt.best_cpu_format.upper())
            parts.append(f"CPU → {tag} {ver}".strip())
        convert_tip = "Convert: " + " + ".join(parts) if parts else ""

        any_eligible = False
        for cid in TARGET_CLASS_IDS:
            path = self._state.model_paths.get(cid)
            is_pt = bool(path and path.lower().endswith(".pt"))

            if not is_pt:
                self._mconv[cid].setVisible(False)
                continue

            # Per-device existence checks
            gpu_done = (not has_cuda) or self._optimized_exists_for(cid, "cuda")
            cpu_done = self._optimized_exists_for(cid, "cpu")
            all_done = gpu_done and cpu_done
            no_runtime = not parts  # no useful format for any device

            btn = self._mconv[cid]
            if all_done:
                # Both devices optimised — green check
                gpu_fmt = {"engine": "TRT", "onnx": "ONNX"}.get(rt.best_gpu_format, "")
                cpu_fmt = {"openvino": "OV",  "onnx": "ONNX"}.get(rt.best_cpu_format, "")
                check_tip = " + ".join(filter(None, [
                    (f"GPU:{gpu_fmt}" if has_cuda else ""),
                    f"CPU:{cpu_fmt}",
                ]))
                btn.setVisible(True)
                btn.setEnabled(False)
                btn.setIcon(_fa("check-circle", "#4CAF50", int(11 * _S)))
                btn.setToolTip(f"Fully optimised ({check_tip})")
            elif no_runtime:
                btn.setVisible(True)
                btn.setEnabled(False)
                btn.setIcon(_fa("exclamation-triangle", "#888", int(11 * _S)))
                btn.setToolTip("No optimisation runtime installed (tensorrt / openvino)")
            else:
                # At least one device still needs conversion
                missing = []
                if has_cuda and not gpu_done:
                    missing.append("GPU")
                if not cpu_done:
                    missing.append("CPU")
                pending_tip = convert_tip + (f" [{', '.join(missing)} pending]" if missing else "")
                btn.setVisible(True)
                btn.setEnabled(True)
                btn.setIcon(_fa("bolt", _TD, int(11 * _S)))
                btn.setToolTip(pending_tip)
                any_eligible = True

        self.btnOptimizeAll.setVisible(any_eligible)

    def _build_convert_jobs(self, cids=None):
        """Return a list of ConvertJob tuples for eligible models.

        Always queues jobs for BOTH devices so one click optimises for
        GPU (TensorRT) AND CPU (OpenVINO) simultaneously.
        """
        if cids is None:
            cids = TARGET_CLASS_IDS
        imgsz = self._state.get_imgsz()
        # Always convert for both targets; skip devices that have no useful format
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        jobs = []
        for dev in devices:
            half = self._state.use_fp16() and dev == "cuda"
            for cid in cids:
                path = self._state.model_paths.get(cid)
                if not path or not path.lower().endswith(".pt"):
                    continue
                if self._optimized_exists_for(cid, dev):
                    continue
                jobs.append((cid, path, dev, imgsz, half))
        return jobs

    def _on_convert_model(self, cid: int) -> None:
        """Start conversion for a single model (GPU + CPU)."""
        jobs = self._build_convert_jobs(cids=[cid])
        if not jobs:
            self._set_status("Nothing to convert (already optimised or not a .pt model)")
            return
        rt = _detect_runtimes()
        parts = []
        if torch.cuda.is_available():
            parts.append(rt.gpu_summary)
        parts.append(rt.cpu_summary)
        self._set_status("Starting conversion · " + " | ".join(parts))
        self._start_conversion(jobs)

    def _on_convert_all(self) -> None:
        """Start conversion for every eligible model (GPU + CPU)."""
        jobs = self._build_convert_jobs()
        if not jobs:
            self._set_status("All models already optimised")
            return
        rt = _detect_runtimes()
        parts = []
        if torch.cuda.is_available():
            parts.append(rt.gpu_summary)
        parts.append(rt.cpu_summary)
        self._set_status("Batch conversion started · " + " | ".join(parts))
        self._start_conversion(jobs)

    def _start_conversion(self, jobs) -> None:
        if self._converter and self._converter.isRunning():
            self._set_status("A conversion is already running — please wait")
            return
        self._converter = ConvertWorker(jobs)   # no parent — we manage lifetime
        self._converter.conversion_started.connect(self._on_cvt_started)
        self._converter.conversion_progress.connect(self._on_cvt_progress)
        self._converter.conversion_finished.connect(self._on_cvt_finished)
        self._converter.all_finished.connect(self._on_cvt_all_done)
        # Disable convert buttons while running
        for cid in TARGET_CLASS_IDS:
            self._mconv[cid].setEnabled(False)
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
                f"{CLASS_NAMES.get(cid, '')} → {fmt_name} ✓  ({os.path.basename(result)})")
            btn = self._mconv.get(cid)
            if btn:
                btn.setIcon(_fa("check-circle", "#4CAF50", int(11 * _S)))
                btn.setToolTip(f"Optimised ({fmt_name})")
                btn.setEnabled(False)
        else:
            self._set_status(
                f"{CLASS_NAMES.get(cid, '')} conversion failed: {result.splitlines()[0]}")
            btn = self._mconv.get(cid)
            if btn:
                btn.setIcon(_fa("exclamation-triangle", "#f44336", int(11 * _S)))
                btn.setToolTip(f"Conversion failed — click to retry")
                btn.setEnabled(True)

    def _on_cvt_all_done(self) -> None:
        """All jobs finished — refresh buttons and trigger model hot-reload."""
        self._cvt_timer.stop()
        self._update_convert_buttons()
        for cid in TARGET_CLASS_IDS:
            self._mconv[cid].setEnabled(True)
        self.btnOptimizeAll.setEnabled(True)
        # Tell inference engine to reload with the new optimised weights
        self._state.reload_models_flag = True
        self._set_status("Model optimisation complete — inference will reload")

    def _show_speed_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(f"""
            QMenu{{background:#161616;color:{_T};
                border:1px solid {_BD};padding:{int(4*_S)}px 0;}}
            QMenu::item{{padding:{int(5*_S)}px {int(20*_S)}px;
                font-size:{int(11*_S)}px;
                font-family:"Consolas","Courier New",monospace;}}
            QMenu::item:selected{{background:#2a2a2a;}}""")
        for i, (label, rate) in enumerate(self._speeds):
            act = menu.addAction(label); act.setData((i, rate))
            if i == self._spd_idx:
                act.setCheckable(True); act.setChecked(True)
        pos = self.btnSpeed.mapToGlobal(QtCore.QPoint(0, 0))
        pos.setY(pos.y() - menu.sizeHint().height())
        chosen = menu.exec_(pos)
        if chosen:
            i, r = chosen.data()
            self._spd_idx = i
            self._state.set_playback_rate(float(r))

    # ── helpers ───────────────────────────────────────────────────────────

    def _init_seek(self) -> None:
        if self._total_frames > 1:
            self.seekSlider.setEnabled(True)
            self.seekSlider.setRange(0, self._total_frames - 1)
        else:
            self.seekSlider.setEnabled(False)
            self.seekSlider.setRange(0, 0)
        if self._video_fps > 0 and self._total_frames > 0:
            self.lblTotal.setText(
                self._fmt((self._total_frames - 1) / self._video_fps))

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int((sec or 0) + 0.5))
        return f"{sec // 60}:{sec % 60:02d}"

    def _set_status(self, t: str) -> None:
        self.statusLabel.setText(t)

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

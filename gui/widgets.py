"""
Custom PyQt5 widgets — drag-and-drop labels + seek slider.
"""

import os
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class SeekSlider(QtWidgets.QSlider):
    """
    Horizontal seek slider that always draws a white circle at the
    current handle position (YouTube / Miruro style).  The circle
    grows slightly when the mouse is over the widget.
    """

    def __init__(self, scale: float = 1.0,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(QtCore.Qt.Horizontal, parent)
        # reduce the indicator sizes so the circle is smaller on hover
        self._dot_r   = max(1, round(4 * scale))   # normal radius
        self._dot_r_h = max(1, round(6 * scale))   # hovered radius
        self._hovered = False
        self.setMouseTracking(True)

    # ── hover tracking ────────────────────────────────────────────────────

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.isEnabled()
            and self.maximum() > self.minimum()
        ):
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                opt,
                QtWidgets.QStyle.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                opt,
                QtWidgets.QStyle.SC_SliderHandle,
                self,
            )
            if not handle.contains(event.pos()) and groove.width() > 1:
                pos = max(0, min(groove.width(), event.pos().x() - groove.x()))
                value = QtWidgets.QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    pos,
                    groove.width(),
                    upsideDown=False,
                )
                self.setValue(value)
                self.sliderMoved.emit(value)
                self.sliderPressed.emit()
                self.sliderReleased.emit()
                event.accept()
                return
        super().mousePressEvent(event)

    # ── custom paint ──────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        # Draw groove + progress via stylesheet (handle is transparent in CSS)
        super().paintEvent(event)

        if not self._hovered:
            return
        if not self.isEnabled() or self.maximum() == self.minimum():
            return

        # Ask Qt where the handle centre is
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        handle_rect = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider,
            opt,
            QtWidgets.QStyle.SC_SliderHandle,
            self,
        )
        cx = handle_rect.center().x()
        cy = self.height() // 2

        r = self._dot_r_h

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#ffffff"))
        p.drawEllipse(QtCore.QPoint(cx, cy), r, r)
        p.end()


class DropLabel(QtWidgets.QLabel):
    """QLabel that accepts a single file via drag-and-drop."""

    fileDropped = QtCore.pyqtSignal(str)

    def __init__(
        self,
        text: str = "",
        allowed_exts: Tuple[str, ...] = (),
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self._exts = tuple(e.lower() for e in allowed_exts)
        self.setAcceptDrops(True)

    def _ok(self, path: str) -> bool:
        if not self._exts:
            return True
        return os.path.splitext(path)[1].lower() in self._exts

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            if len(paths) == 1 and self._ok(paths[0]):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if len(paths) == 1 and self._ok(paths[0]):
            self.fileDropped.emit(paths[0])
            event.acceptProposedAction()
        else:
            event.ignore()


class VideoDropLabel(QtWidgets.QLabel):
    """Large video display area that accepts video + model file drops."""

    filesDropped = QtCore.pyqtSignal(list)

    def __init__(self, text: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

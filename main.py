"""
Safety Gear Compliance – Multi-Threaded Pipeline
=================================================
Entry point.  Run with::

    python main.py
"""

import os
import sys

# Silence FFmpeg's noisy "reference picture missing during reorder" / H.264
# decoder warnings that occur after every random-access seek.  Must be set
# BEFORE OpenCV (and therefore the FFmpeg backend) is imported anywhere.
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"          # = quiet
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"             # suppress dshow probe
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "loglevel;quiet|threads;auto",
)
# Suppress TensorRT [I]/[W] boot messages (engine size, MemUsage, logger warn).
os.environ.setdefault("TRT_LOGGER_SEVERITY", "3")

from PyQt5 import QtCore, QtWidgets  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402


def main() -> int:
    # Enable high-DPI scaling before QApplication is created.
    # Use getattr so the script works on Qt builds where the attribute
    # was removed (Qt 6) and silences static type-checkers on Qt 5 stubs.
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    for _attr in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        flag = getattr(QtCore.Qt, _attr, None)
        if flag is not None:
            QtWidgets.QApplication.setAttribute(flag, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

"""
Safety Gear Compliance – Multi-Threaded Pipeline
=================================================
Entry point.  Run with::

    python main.py
"""

import os
import sys
from PyQt5 import QtCore, QtWidgets
from gui.main_window import MainWindow


def main() -> int:
    # Enable high-DPI scaling before QApplication is created
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

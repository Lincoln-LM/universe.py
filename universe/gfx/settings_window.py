"""Separate QT Settings Window"""

from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING
from math import log2

os.environ["QT_API"] = "pyside6"
# pylint: disable=wrong-import-position,no-name-in-module
import qdarkstyle  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
)

# pylint: enable=wrong-import-position,no-name-in-module

if TYPE_CHECKING:
    from .main_window import MainWindow


class SettingsWindow(QWidget):
    """Separate QT Settings Window"""

    SPEEDS = (
        ("1s", 1),
        ("5s", 5),
        ("30s", 30),
        ("60s", 60),
        ("5m", 60 * 5),
        ("15m", 60 * 15),
        ("30m", 60 * 30),
        ("1h", 60 * 60),
        ("2h", 60 * 60 * 2),
        ("5h", 60 * 60 * 5),
        ("12h", 60 * 60 * 12),
        ("1d", 60 * 60 * 24),
        ("3d", 60 * 60 * 24 * 3),
        ("1w", 60 * 60 * 24 * 7),
        ("2w", 60 * 60 * 24 * 7 * 2),
        ("1m", 60 * 60 * 24 * 7 * 4),
        ("2mo", 60 * 60 * 24 * 7 * 4 * 2),
        ("3mo", 60 * 60 * 24 * 7 * 4 * 3),
        ("6mo", 60 * 60 * 24 * 7 * 4 * 6),
        ("1y", 60 * 60 * 24 * 7 * 4 * 12),
        ("2y", 60 * 60 * 24 * 7 * 4 * 12 * 2),
        ("5y", 60 * 60 * 24 * 7 * 4 * 12 * 5),
        ("10y", 60 * 60 * 24 * 7 * 4 * 12 * 10),
    )

    def __init__(self, main_window: MainWindow):
        super().__init__()

        self.main_window = main_window

        self.setWindowTitle("Settings")
        self.main_layout = QVBoxLayout(self)

        self.zoom_slider_label = QLabel("Zoom")

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(10 * self.main_window.universe_camera.max_zoom)

        self.zoom_spinbox = QDoubleSpinBox(
            minimum=1,
            maximum=2**self.main_window.universe_camera.max_zoom,
        )
        self.zoom_spinbox.setValue(self.main_window.zoom_scale)

        self.speed_slider_label = QLabel("Speed")

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(len(self.SPEEDS) - 1)

        self.speed_combobox = QComboBox()
        for item in self.SPEEDS:
            self.speed_combobox.addItem(item[0])

        self.main_layout.addWidget(self.zoom_slider_label)
        self.main_layout.addWidget(self.zoom_slider)
        self.main_layout.addWidget(self.zoom_spinbox)
        self.main_layout.addWidget(self.speed_slider_label)
        self.main_layout.addWidget(self.speed_slider)
        self.main_layout.addWidget(self.speed_combobox)

        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_change)
        self.zoom_spinbox.valueChanged.connect(self.on_zoom_spinbox_change)
        self.speed_slider.valueChanged.connect(self.on_speed_slider_change)
        self.speed_combobox.currentIndexChanged.connect(self.on_speed_combobox_change)

        self.value_changing = False

    def on_zoom_slider_change(self):
        """Callback for when the zoom slider changes"""
        if self.value_changing:
            return
        self.value_changing = True
        self.main_window.universe_camera.new_zoom = self.zoom_slider.value() / 10
        self.main_window.update_trail_size()
        self.zoom_spinbox.setValue(self.main_window.zoom_scale)
        self.value_changing = False

    def on_zoom_spinbox_change(self):
        """Callback for when the zoom spinbox changes"""
        if self.value_changing:
            return
        self.value_changing = True
        self.main_window.universe_camera.new_zoom = -log2(
            self.zoom_spinbox.value() / self.main_window.scale
        )
        self.main_window.update_trail_size()
        self.zoom_slider.setValue(self.main_window.universe_camera.new_zoom * 10)
        self.value_changing = False

    def on_speed_slider_change(self):
        """Callback for when the speed slider changes"""
        if self.value_changing:
            return
        self.value_changing = True
        self.main_window.universe.time_scale = self.SPEEDS[self.speed_slider.value()][1]
        self.speed_combobox.setCurrentIndex(self.speed_slider.value())
        self.value_changing = False

    def on_speed_combobox_change(self, new_index: int):
        """Callback for when the speed combobox changes

        Args:
            new_index (int): New selected index
        """
        if self.value_changing:
            return
        self.value_changing = True
        self.main_window.universe.time_scale = self.SPEEDS[new_index][1]
        self.speed_slider.setValue(new_index)
        self.value_changing = False


class SettingsWindowThread(threading.Thread):
    """Thread for the settings window to run seperately"""

    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window
        self.gui: SettingsWindow = None

    def run(self):
        app = QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet())

        self.gui = SettingsWindow(self.main_window)
        self.gui.show()

        app.exec_()

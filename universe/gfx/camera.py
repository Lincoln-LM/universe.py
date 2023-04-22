"""A 2D camera for the universe scene"""

from __future__ import annotations
from typing import TYPE_CHECKING
from math import log2

if TYPE_CHECKING:
    from .main_window import MainWindow


class Camera:
    """A 2D camera for the universe scene"""

    def __init__(
        self,
        window: MainWindow,
    ) -> None:
        self._window = window
        self.max_zoom = log2(self._window.scale)
        self._zoom = 1
        self.new_zoom = self._zoom

    @property
    def zoom(self) -> int:
        """Current zoom property"""
        return self._zoom

    @zoom.setter
    def zoom(self, value: int) -> None:
        self._zoom = max(min(value, self.max_zoom), 1)

    @property
    def zoom_scale(self) -> int:
        """Zoom pixel scale"""
        return self._window.scale / (2**self.zoom)

    @property
    def zoom_scale_inv(self) -> int:
        """Inverse zoom pixel scale"""
        return 1 / self.zoom_scale

    def begin(self) -> None:
        """Begin camera instance"""
        center_ofs = self._window.width // 2, self._window.height // 2

        # Move origin to the center of the screen
        view_matrix = self._window.view.translate(
            (
                center_ofs[0],
                center_ofs[1],
                0,
            )
        )
        # Zoom out to scale
        view_matrix = view_matrix.scale((self.zoom_scale_inv, self.zoom_scale_inv, 1))
        if self._window.target is not None:
            # Move to target
            view_matrix = view_matrix.translate(
                (
                    -self._window.target.position[0],
                    -self._window.target.position[1],
                    0,
                )
            )
        self._window.view = view_matrix

    def end(self) -> None:
        """End camera instance"""
        center_ofs = self._window.width // 2, self._window.height // 2

        view_matrix = self._window.view
        if self._window.target is not None:
            # Move away from target
            view_matrix = view_matrix.translate(
                (
                    self._window.target.position[0],
                    self._window.target.position[1],
                    0,
                )
            )
        # Zoom back in
        view_matrix = view_matrix.scale(
            (
                1 / (self.zoom_scale_inv),
                1 / (self.zoom_scale_inv),
                1,
            )
        )
        # Move origin to bottom left
        self._window.view = view_matrix.translate(
            (
                -center_ofs[0],
                -center_ofs[1],
                0,
            )
        )

    def __enter__(self) -> None:
        self.begin()

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        self.end()

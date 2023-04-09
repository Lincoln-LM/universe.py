"""A 2D camera for the universe scene"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainWindow


class Camera:
    """A 2D camera for the universe scene"""

    def __init__(
        self,
        window: MainWindow,
        scroll_speed=1,
        min_zoom=1,
        max_zoom=100,
        target=None,
    ) -> None:
        assert (
            min_zoom <= max_zoom
        ), "Minimum zoom must not be greater than maximum zoom"
        self._window = window
        self.scroll_speed = scroll_speed
        self.max_zoom = max_zoom
        self.min_zoom = min_zoom
        self.target = target
        self._zoom = max(min(1, self.max_zoom), self.min_zoom)

    @property
    def zoom(self) -> int:
        """Current zoom property"""
        return self._zoom

    @zoom.setter
    def zoom(self, value: int) -> None:
        self._zoom = max(min(value, self.max_zoom), self.min_zoom)

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
        # Move to target
        self._window.view = view_matrix.translate(
            (
                -self.target.position[0],
                -self.target.position[1],
                0,
            )
        )

    def end(self) -> None:
        """End camera instance"""
        center_ofs = self._window.width // 2, self._window.height // 2

        # Move away from target
        view_matrix = self._window.view.translate(
            (
                self.target.position[0],
                self.target.position[1],
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

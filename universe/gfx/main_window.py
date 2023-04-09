"""Main GUI window for universe.py"""

import pyglet
from .camera import Camera
from ..obj.massive_body import MassiveBody
from ..obj.universe import Universe


class MainWindow(pyglet.window.Window):
    """Main GUI window for universe.py"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale = 384400000 / (self.width * 0.5 * 0.75)

        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)

        self.universe_camera = Camera(self)

        self.universe = Universe()
        self.gfx_objects: list[MassiveBody] = []
        self.trail_objects = []
        self.max_trail = 1000

        self.running = False

        self.event(self.on_draw)
        pyglet.clock.schedule(self.update)

    @property
    def zoom_scale(self) -> int:
        """Zoom pixel scale"""
        return self.scale / self.universe_camera.zoom

    # pylint: disable=arguments-differ
    def on_draw(self):
        """Redraw all rendered objects"""
        self.clear()
        if self.running:
            for body in self.gfx_objects:
                self.trail_objects.append(body.draw_trail(self))
                # discard any trail objects created before limit
                # automatically unrenders/handles deletion
                self.trail_objects = self.trail_objects[-self.max_trail :]
                body.update_gfx(self)
            with self.universe_camera:
                self.batch.draw()
        self.fps_display.draw()

    # pylint: enable=arguments-differ

    def on_key_release(self, symbol: int, _modifiers: int):
        if symbol == pyglet.window.key.ENTER:
            self.running = True

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.universe_camera.zoom += scroll_y

    def update(self, delta_time: float) -> None:
        """Update all objects each tick

        Args:
            delta_time (float): Delta time elapsed since the last tick
        """
        if self.running:
            self.universe.update(delta_time)

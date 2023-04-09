"""Main GUI window for universe.py"""

import pyglet
from .camera import Camera
from ..obj.massive_body import MassiveBody
from ..obj.universe import Universe


class MainWindow(pyglet.window.Window):
    """Main GUI window for universe.py"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale = 149597870700

        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.zoom_label = pyglet.text.Label(
            "",
            font_size=12,
            x=self.width,
            y=self.height,
            anchor_x="right",
            anchor_y="top",
        )
        self.target_label = pyglet.text.Label(
            "",
            font_size=12,
            x=0,
            y=self.height,
            anchor_x="left",
            anchor_y="top",
        )

        self.universe_camera = Camera(self)
        self.target = None
        self.target_index = -1

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
        return self.universe_camera.zoom_scale

    @property
    def zoom_scale_inv(self) -> int:
        """Inverse zoom pixel scale"""
        return self.universe_camera.zoom_scale_inv

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
        if self.target is not None:
            self.target_label.text = f"Target: {self.target.name}"
        else:
            self.target_label.text = "Target: None"
        self.target_label.draw()
        self.zoom_label.text = f"M/pixel: {self.zoom_scale:.00f}"
        self.zoom_label.draw()

    # pylint: enable=arguments-differ

    def on_key_release(self, symbol: int, _modifiers: int):
        if symbol == pyglet.window.key.ENTER:
            self.running = True
        elif symbol == pyglet.window.key.TAB:
            self.target_index += 1
            if self.target_index >= len(self.gfx_objects):
                self.target_index = -1
                self.target = None
            else:
                self.target = self.gfx_objects[self.target_index]

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.universe_camera.zoom += scroll_y
        scale = self.zoom_scale
        for trail_object in self.trail_objects:
            trail_object.radius = scale

    def update(self, delta_time: float) -> None:
        """Update all objects each tick

        Args:
            delta_time (float): Delta time elapsed since the last tick
        """
        if self.running:
            self.universe.update(delta_time)

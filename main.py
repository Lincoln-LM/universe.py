"""Example usage script"""

import pyglet
from universe.gfx.main_window import MainWindow
from universe.obj.massive_body import MassiveBody

window = MainWindow(500, 500)

# sun
sun = MassiveBody(1.9891 * 10**30, 695700000, (0, 0), (0, 0))
sun.py_init(window, "Sun")


# earth
earth = MassiveBody(5.9722 * 10**24, 6378137, (0, 149597870700), (29780, 0))
earth.py_init(window, "Earth")

# moon
moon = MassiveBody(
    0.07346 * 10**24,
    1738100,
    (earth.position[0] + 384400000, earth.position[1]),
    (earth.velocity[0], earth.velocity[1] - 1022),
)
moon.py_init(window, "Moon")

window.universe_camera.target = sun

pyglet.app.run()

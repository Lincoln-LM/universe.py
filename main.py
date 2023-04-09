"""Example usage script"""

import pyglet
from universe.gfx.main_window import MainWindow
from universe.obj.massive_body import MassiveBody

window = MainWindow(500, 500)

# earth
parent = MassiveBody(5.9722 * 10**24, 6378137, (0, 0), (0, 0))
parent.py_init(window)

# moon
child = MassiveBody(0.07346 * 10**24, 1738100, (384400000, 0), (0, -1022))
child.py_init(window)

window.universe_camera.target = child

pyglet.app.run()

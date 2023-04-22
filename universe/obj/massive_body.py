"""Object with enough mass to create a gravitational field"""

from __future__ import annotations
import typing
import numba
import numpy as np
import pyglet
from ..constants import G
from ..partial_jitclass import partial_jitclass, njit_spec, py_func, INSTANCE_TYPE

if typing.TYPE_CHECKING:
    from ..gfx.main_window import MainWindow
    from .universe import Universe


@partial_jitclass
class MassiveBody:
    """Object with enough mass to create a gravitational field"""

    # numba attributes
    mass: np.float64
    radius: np.float64
    position: np.ndarray[np.float64]
    velocity: np.ndarray[np.float64]
    acceleration: np.ndarray[np.float64]
    id: np.int64

    # python attributes
    __slots__ = ("shape", "name")
    shape: pyglet.shapes.Circle
    name: str

    @njit_spec(
        numba.types.none(
            numba.float64,
            numba.float64,
            numba.types.UniTuple(numba.float64, 2),
            numba.types.UniTuple(numba.float64, 2),
        )
    )
    def __init__(
        self,
        mass: np.float64,
        radius: np.float64,
        position: tuple[np.float64, np.float64],
        velocity: tuple[np.float64, np.float64],
    ) -> None:
        self.mass = mass
        self.radius = radius
        self.position = np.empty(2, np.float64)
        self.position[:] = position
        self.velocity = np.empty(2, np.float64)
        self.velocity[:] = velocity
        self.acceleration = np.zeros(2, np.float64)
        self.id = -1

    @njit_spec(numba.none(numba.float64))
    def update_step(self, delta_time: np.float64) -> None:
        """Update object according to a fraction of the progression of time

        Args:
            delta_time (np.float64): The amount of time to process this step
        """
        self.velocity += self.acceleration * delta_time
        self.position += self.velocity * delta_time

        # reset acceleration for next step
        self.acceleration[:] = 0

    @njit_spec(numba.none(numba.float64, numba.bool_))
    def update_half_step(self, delta_time: np.float64, is_first_step: np.bool8) -> None:
        """Update object according to half of a fraction of the progression of time

        Args:
            delta_time (np.float64): The full amount of time to process this step
            is_first_step (np.bool8): Whether or not the half step is the first one
        """
        if not is_first_step:
            self.velocity += self.acceleration * delta_time  # full timestep
        self.position += self.velocity * delta_time / 2  # half timestep

        # reset acceleration for next step
        self.acceleration[:] = 0

    @njit_spec(numba.none(INSTANCE_TYPE))
    def apply_gravity(self, other: MassiveBody) -> None:
        """Apply gravitational acceleration to self

        Args:
            other (MassiveBody): The body pulling on self
        """
        self.acceleration += self.gravitational_acceleration(other)

    @njit_spec(numba.none(INSTANCE_TYPE))
    def apply_mirrored_gravity(self, other: MassiveBody) -> None:
        """Apply gravitational acceleration to both objects

        Args:
            other (MassiveBody): The body pulling on self
        """
        distance_vec = other.position - self.position
        distance_cubed = np.dot(distance_vec, distance_vec) ** (3 / 2)

        force = G / distance_cubed * distance_vec

        self.acceleration += other.mass * force
        other.acceleration -= self.mass * force

    @njit_spec(numba.float64[::1](INSTANCE_TYPE))
    def gravitational_acceleration(self, other: MassiveBody) -> np.ndarray[np.float64]:
        """Compute the gravitational acceleration

        Args:
            other (MassiveBody): The body pulling on self

        Returns:
            np.ndarray[np.float64]: The acceleration vector caused by the pull from other
        """
        distance_vec = other.position - self.position
        distance_cubed = np.dot(distance_vec, distance_vec) ** (3 / 2)

        return G * other.mass / distance_cubed * distance_vec

    @py_func
    def update_gfx(self, window: MainWindow) -> None:
        """Update pyglet graphics

        Args:
            window (MainWindow): Parent window
        """
        position = self.position_tuple()
        self.shape.position = (
            position[0],
            position[1],
        )
        self.shape.radius = max(
            self.radius,
            5 * window.zoom_scale,
        )

    @py_func
    def draw_trail(self, window: MainWindow) -> pyglet.shapes.Circle:
        """Draw small trail at the position of the object

        Args:
            window (MainWindow): Parent window

        Returns:
            pyglet.shapes.Circle: New trail object
        """
        trail = pyglet.shapes.Circle(*self.shape.position, 1, batch=window.batch)
        trail.radius = window.zoom_scale
        return trail

    @py_func
    def py_init(
        self,
        universe: Universe,
        window: MainWindow = None,
        name: str = "Unnamed Massive Body",
    ) -> None:
        """Initializer for interpreter-only attributes

        Args:
            universe (Universe): Containing universe
            window (MainWindow): Parent window
            name (str): The name of the object
        """
        universe.add_massive_body(self)
        self.id = universe.current_id
        # increment to ensure each object gets a unique id
        # 2^63 should be enough
        universe.current_id += 1
        self.name = name
        if window is not None:
            self.shape = pyglet.shapes.Circle(
                *self.position_tuple(), 1, batch=window.batch
            )
            self.update_gfx(window)
            window.gfx_objects.append(self)

    @py_func
    def position_tuple(self) -> tuple[np.float64, np.float64]:
        """Get a copy of the X,Y position as a tuple

        Returns:
            tuple[np.float64, np.float64]: The X,Y position as a tuple
        """
        return (self.position[0], self.position[1])

"""Collection of all objects in a universe"""

import numba

# pylint: disable=no-name-in-module
from numba.typed import List

# pylint: enable=no-name-in-module
import numpy as np
from .massive_body import MassiveBody
from ..partial_jitclass import partial_jitclass, njit_spec
from ..partial_jitclass.base import convert_to_numba

MAX_DELTA_TIME = 60 * 60

MASSIVE_BODY_INSTANCE_TYPE = convert_to_numba(MassiveBody)


@partial_jitclass
class Universe:
    """Collection of all objects in a universe"""

    massive_bodies: list[MassiveBody]
    current_id: np.int64
    time_scale: np.float64

    def __init__(self) -> None:
        self.massive_bodies = List.empty_list(MASSIVE_BODY_INSTANCE_TYPE)
        self.current_id = 0
        self.time_scale = 1

    @njit_spec(numba.none(numba.float64))
    def update(self, delta_time: np.float64) -> None:
        """Update object according to the progression of time

        Args:
            delta_time (np.float64): The total amount of time to progress
        """
        delta_time *= self.time_scale
        for step in np.arange(0, delta_time + MAX_DELTA_TIME, MAX_DELTA_TIME):
            step = delta_time - step
            if step > 0:
                step = MAX_DELTA_TIME
            for massive_body in self.massive_bodies:
                massive_body.update_half_step(step)
            total_bodies = len(self.massive_bodies)
            for i in range(total_bodies):
                for j in range(i + 1, total_bodies):
                    self.massive_bodies[i].apply_mirrored_gravity(
                        self.massive_bodies[j]
                    )
            for massive_body in self.massive_bodies:
                massive_body.update_half_step(step)

    @njit_spec(numba.none(MASSIVE_BODY_INSTANCE_TYPE))
    def add_massive_body(self, massive_body: MassiveBody) -> None:
        """Add a massive body to the universe

        Args:
            massive_body (MassiveBody): Massive body to add
        """
        self.massive_bodies.append(massive_body)

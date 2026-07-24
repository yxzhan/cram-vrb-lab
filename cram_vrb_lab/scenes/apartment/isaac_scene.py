"""Apartment scene loading on the Isaac Sim side.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim

from .constants import APARTMENT_USD_PATH, GRID_USD_PATH, USD_PRIM_POSITION_IN_MAP


def load_apartment_scene(world, render):
    """Ground grid, the apartment USD, lights, and the default camera view."""
    # Ground
    define_prim("/World/Ground", "Xform").GetReferences().AddReference(GRID_USD_PATH)

    # Apartment
    create_prim(
        usd_path=APARTMENT_USD_PATH,
        prim_path="/World/Apartment",
        position=np.array(USD_PRIM_POSITION_IN_MAP),
    )

    # Lights so the raytraced scene is not black
    for i in range(0, 4):
        create_prim(
            prim_path=f"/World/Ground/Light_{i}",
            prim_type="SphereLight",
            attributes={"inputs:intensity": 10000},
            position=(2 - 4 * i, 0, 2),
        )

    viewports.set_camera_view(eye=np.array([-6.5, -2, 2]), target=np.array([-1, 1, 1]))

    for _ in range(30):
        world.step(render=render)

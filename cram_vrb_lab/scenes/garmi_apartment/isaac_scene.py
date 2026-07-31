"""garmi-apartment scene loading on the Isaac Sim side.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim
from pxr import UsdGeom

from .constants import (
    GARMI_APARTMENT_USD_PATH,
    GRID_USD_PATH,
    USD_PRIM_POSITION_IN_MAP,
    STRETCH_SPAWN_POSITION
)

def load_garmi_apartment_scene(world, render, camera_eye=None, camera_target=None):
    """Ground grid, the apartment USD, fill lights, and the default camera view.

    :param camera_eye: viewport camera position; defaults to the view saved in
        ``world.usda``'s own ``customLayerData``, which frames the living room.
    :param camera_target: what the viewport looks at.
    """
    # Ground
    define_prim("/World/Ground", "Xform").GetReferences().AddReference(GRID_USD_PATH)
    UsdGeom.Imageable(
        world.stage.GetPrimAtPath("/World/Ground")
    ).MakeInvisible()

    # Apartment. Placed at the origin so the render and the MJCF twin share one
    # coordinate frame; see constants.USD_PRIM_POSITION_IN_MAP.
    create_prim(
        usd_path=GARMI_APARTMENT_USD_PATH,
        prim_path="/World/GarmiApartment",
        position=np.array(USD_PRIM_POSITION_IN_MAP),
    )

    # Defaults lifted from world.usda's saved Perspective camera, i.e. the view
    # the scene was authored from: over the robot's shoulder into the living room.
    viewports.set_camera_view(
        eye=np.array(
            camera_eye if camera_eye is not None else [-1.0, 2.0, 1.5]
        ),
        target=np.array(
            camera_target if camera_target is not None else [1, 6.0, 0.8]
        ),
    )

    for _ in range(30):
        world.step(render=render)

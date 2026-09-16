"""Let this repo keep talking numpy to Isaac while Newton runs on the GPU.

Isaac picks the array type its wrappers speak from the physics device: CPU
physics keeps numpy, a GPU pipeline forces torch ("Changing backend from 'numpy'
to 'torch' since NumPy cannot be used with GPU piplines"). Newton is a GPU engine
-- on the CPU it runs this scene at a fifth of real time -- so every numpy array
the robots hand to ``Articulation`` and ``XFormPrim`` becomes a type error, of
the unhelpful kind::

    AttributeError: 'numpy.ndarray' object has no attribute 'detach'

Rather than rewrite forty-odd call sites in a dialect that is wrong again the
moment the engine changes, :func:`install` converts at the one place they all go
through. Isaac's prim wrappers do not touch the physics view with what they are
given: they pass it through their backend-utils module first (``move_data``,
``assign``, ``resolve_indices``, ...), and that module is handed to them by
``SimulationManager._get_backend_utils``. Wrapping it so every numpy argument
arrives as a torch tensor is one patch, and prims built afterwards pick it up.

Outputs are untouched: they come back as torch tensors, which
:func:`~cram_vrb_lab.sim.ros_utils.as_np` already normalises -- it was written
for warp arrays and answers a tensor the same way.
"""

import numpy as np


def install():
    """Make Isaac's torch backend accept numpy. Idempotent; returns whether it did.

    A no-op unless the backend actually is torch, so a PhysX run (numpy backend)
    is left exactly as it was.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_backend() != "torch":
        return False
    if getattr(SimulationManager._get_backend_utils, "_numpy_bridged", False):
        return True

    original = SimulationManager._get_backend_utils

    def _get_backend_utils():
        utils = original()
        if SimulationManager.get_backend() != "torch":
            return utils
        return _NumpyTolerant(utils)

    _get_backend_utils._numpy_bridged = True
    SimulationManager._get_backend_utils = _get_backend_utils
    return True


#: Backend-utils functions Isaac asks for host memory from, because PhysX keeps
#: its drive parameters there. Newton keeps them on the GPU, so answering "cpu"
#: literally lands a host tensor next to a device one::
#:
#:     stiffnesses = self._backend_utils.assign(kps, stiffnesses, [...])
#:     RuntimeError: Expected all tensors to be on the same device, but found at
#:     least two devices, cuda:0 and cpu!
#:
#: -- with ``kps`` and the indices on the host and the stiffness tensor they are
#: written into on the device. The device the simulation is actually on is the
#: answer to that question for both engines.
_DEVICE_FOLLOWS_SIMULATION = ("move_data", "resolve_indices")


class _NumpyTolerant:
    """A backend-utils module that takes numpy arrays as well as torch tensors."""

    def __init__(self, utils):
        object.__setattr__(self, "_utils", utils)

    def __getattr__(self, name):
        attribute = getattr(self._utils, name)
        if not callable(attribute):
            return attribute
        redirect_device = name in _DEVICE_FOLLOWS_SIMULATION

        def call(*args, **kwargs):
            if redirect_device and kwargs.get("device") == "cpu":
                kwargs = {**kwargs, "device": _simulation_device()}
            return attribute(
                *[_as_tensor(a) for a in args],
                **{key: _as_tensor(value) for key, value in kwargs.items()},
            )

        return call


def _simulation_device():
    """The device the physics is on, as torch spells it."""
    from isaacsim.core.simulation_manager import SimulationManager

    return str(SimulationManager.get_device())


def _as_tensor(value):
    """numpy in, torch out; anything else through unchanged.

    On the simulation's device, not the host: what Isaac does with these is write
    them into tensors that live on the GPU, and a CPU tensor there is
    ``Expected all tensors to be on the same device`` -- for index arrays as much
    as for the data.

    Float arrays are narrowed to float32: numpy defaults to float64 and Isaac's
    tensors are single precision, and mixing the two is a dtype error one frame
    later rather than here.
    """
    if not isinstance(value, np.ndarray):
        return value

    import torch

    if value.dtype == np.float64:
        value = value.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(value)).to(_simulation_device())


def numpy_view(view):
    """Wrap a physics view -- an ``Articulation``, a ``RigidPrim`` -- so its
    getters answer numpy again.

    :func:`install` fixes the way in; this is the way out. On the GPU backend
    ``get_joint_positions`` and friends hand back cuda tensors, and the code that
    reads them is numpy throughout -- ``positions[0, indices] = PARK_CONFIGURATION``
    is a ``TypeError: can't assign a list to a torch.cuda.FloatTensor``.

    Private attributes pass through untouched: ``_physics_view`` is read directly
    by the TF publishers, which already normalise what it returns through
    :func:`~cram_vrb_lab.sim.ros_utils.as_np`.

    A no-op on PhysX, where the backend is numpy and the view is returned as it is.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_backend() != "torch":
        return view
    return _NumpyResults(view)


class _NumpyResults:
    """A view that answers numpy, whatever the backend speaks."""

    def __init__(self, wrapped):
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name):
        attribute = getattr(self._wrapped, name)
        if name.startswith("_") or not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            return _to_numpy(attribute(*args, **kwargs))

        return call

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)


def _to_numpy(value):
    """Tensors to arrays, through the tuples the getters return them in."""
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value

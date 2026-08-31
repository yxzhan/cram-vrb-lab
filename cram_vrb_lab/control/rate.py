"""The rate giskard's QP closes its loop at -- one number, read by both processes.

The sim and the control server are separate processes in separate interpreters
(Isaac python and the CRAM venv), and this number matters to both: the server
configures its QP with it, and the sim has to keep its cycle rate *above* it or
the controller closes its loop on feedback that has not been republished since it
last looked. So it is settled here rather than written into either one.

Chosen per machine, not per demo. The right value is slightly below the rate the
sim actually sustains, which depends on the GPU and on how the sim is displayed
(see ``demos/README.md``) -- so it is an environment variable in the style of
``ISAAC_RENDER`` / ``ISAAC_KITCHEN_PROPS``: a notebook sets it once at the top
and both subprocesses inherit it.

.. note::
   Imported by both the Isaac python and the CRAM venv, so -- like
   :mod:`cram_vrb_lab.setups` -- nothing here may import isaacsim, giskardpy or
   ROS. Plain python only.
"""

import os

CONTROL_HZ_ENV = "GISKARD_CONTROL_HZ"
"""Environment variable overriding :data:`DEFAULT_CONTROL_HZ`."""

DEFAULT_CONTROL_HZ = 10.0
"""Hz the QP loop runs at unless told otherwise.

Below the 25 Hz the sim loop manages headless, and below the ~17 Hz a native
Isaac window on the VNC desktop leaves it, so the controller keeps a margin on
both. That margin is the point: ``target_frequency`` sets ``control_dt`` *and*
the MPC's own time step (``QPControllerConfig.__post_init__`` ties them), so a
rate the sim cannot feed means commands that apply for longer than the MPC
assumed -- which giskardpy's own docs call "almost guaranteeing overshoot or
instability", and which shows up as a robot that shakes instead of tracking.

10 rather than the 15 this ran at before because 15 was measured on an RTX 3080
and left nothing for a weaker GPU, a bigger scene or a window on the desktop.
giskardpy warns below 20 Hz; harmless here, and the warning is about a real
robot's feedback rate rather than a simulator's.

.. note::
   ``prediction_horizon`` is in **steps**, so the lookahead in seconds is
   ``prediction_horizon / control_hz``. ``demos/giskard_server.py`` keeps 15
   steps, which at 10 Hz is 1.5 s rather than the 1.0 s it was at 15 Hz: smoother
   commands, and a controller that starts slowing for a target sooner. Set the
   horizon to match the rate if the old lookahead is wanted back.
"""

MINIMUM_CONTROL_HZ = 5.0
"""Below this the QP is refused rather than run.

Not a tuned floor, a sanity one. ``QPControllerConfig`` warns that too low a rate
runs "into infeasibility issues", and a horizon of 15 steps at 5 Hz is already a
3 s lookahead -- past the point where the controller is planning the motion
rather than tracking it.
"""


def control_hz() -> float:
    """The configured rate [Hz]: :data:`CONTROL_HZ_ENV`, or
    :data:`DEFAULT_CONTROL_HZ`.

    :raises ValueError: if the variable is set to something unusable. Deliberately
        loud -- a typo that silently fell back to the default would be paid for in
        a controller mistuned for the machine, which looks like a physics problem
        rather than a configuration one.
    """
    raw = os.environ.get(CONTROL_HZ_ENV)
    if raw is None or raw == "":
        return DEFAULT_CONTROL_HZ
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{CONTROL_HZ_ENV}={raw!r} is not a number") from None
    if value < MINIMUM_CONTROL_HZ:
        raise ValueError(
            f"{CONTROL_HZ_ENV}={raw} is below the {MINIMUM_CONTROL_HZ} Hz floor"
        )
    return value

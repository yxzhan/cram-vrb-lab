"""Ghost hands: a viewer's VR controllers in the Isaac scene, able to drag anything.

A viewer in a headset has a pair of hands in the render that obey no physics -- nothing
pushes them, nothing stops them, they go through the table -- and that can take hold of
any dynamic body there: an object on the worktop, a drawer, a door. What they hold
still obeys physics: it is pulled towards the hand, not put there, so it stops at the
table, knocks what it hits, drags a drawer along its runners and flies on when let go.
What Isaac's own viewport does on shift-drag, from a controller.

The poses come from where cramera's viewer already reports them -- every headset posts
its head and hands to the live bridge (``POST /avatar``) -- together with whether each
hand's trigger is held for a ghost grab. The demo forwards them on
:data:`GHOST_HANDS_TOPIC`; the sim (:class:`~cram_vrb_lab.sim.ghost_hands_bridge.GhostHandsROS`)
does the rest.

Plain data only: both the Isaac python and the CRAM venv import this module.
"""

from __future__ import annotations

GHOST_HANDS_TOPIC = "/cram_vrb_lab/ghost_hands"
"""``std_msgs/String``, JSON ``{"hands": {key: {"position": [x, y, z],
"orientation": [x, y, z, w], "grab": bool, "scale": [x, y, z], "bar": [l, w, h],
"color": "#rrggbb"}}, "gone": [key, ...]}`` in ``map``. A key names one hand of one
viewer (``"<viewer>/left"``); ``gone`` lists hands to take out. ``scale``, ``bar`` and
``color`` are how cramera draws the hand -- a ball of diameter ``scale`` on the hand,
a bar of extents ``bar`` running back from it along -X, in the viewer's colour -- so
the ghost in the render is the same hand the viewers see."""


def hand_key(viewer: str, hand: str) -> str:
    """The key :data:`GHOST_HANDS_TOPIC` names ``viewer``'s ``hand`` by."""
    return f"{viewer}/{hand}"

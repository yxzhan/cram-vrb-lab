"""Where the time goes between a hand in the viewer and the robot, and back.

Two paths are timed, both on this machine's wall clock, which every process on it
shares:

- **Feedback** -- the robot's state on its way to the viewer: the sim publishes a joint
  state, giskard takes it and republishes its world on ``/world_sync``, the twin applies
  that and the live bridge snapshots it, and a viewer reads it (pushed on ``/ws``, or
  polled with ``GET /state`` while the socket is down).
  Correlated by value: while the traced joint moves, every value a later stage shows is
  looked up in the sim's own recent history, and the stage's latency is how long ago the
  sim published it. A joint at rest shows nothing, since any old sample would match.
- **Command** -- a drag's way to the robot: the bridge takes a ``POST /move`` for a
  teleop marker, the demo applies it to the twin, giskard answers with a velocity
  command, and the arm starts moving in the sim. Timed at the start of every drag -- the
  first move after a second of none -- since a continuous drag has no one event to
  follow through.

Every :data:`REPORT_PERIOD` a summary line per path is printed; each drag onset prints
its own line as soon as the arm moves.

Armed from the teleop demo by ``TELEOP_LATENCY_TRACE=1``. Diagnostic only: it wraps a
few of the bridge's methods and adds subscriptions on the demo's node, and changes
nothing on the way through.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

REPORT_PERIOD = 5.0
"""[s] between summary lines."""

HISTORY = 10.0
"""[s] of the sim's joint states kept to look values up in."""

MATCH = 2e-4
"""[rad] how close a stage's value has to be to the sim's to be the same sample."""

MOVING = 1e-4
"""[rad] between consecutive sim samples for the joint to count as moving."""

DRAG_GAP = 1.0
"""[s] without a move before the next one starts a drag."""

ONSET_TIMEOUT = 5.0
"""[s] after a drag starts before giving up on the arm moving."""


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.0f}"


def _summary(values: List[float]) -> str:
    if not values:
        return "--"
    array = np.asarray(values)
    return (f"med {_ms(np.median(array))} p95 {_ms(np.percentile(array, 95))} "
            f"max {_ms(array.max())} ms (n={len(array)})")


class LatencyTrace:
    """Times both paths; see the module docstring.

    :param node: the demo's rclpy node, already spun.
    :param world: the twin.
    :param bridge: the live bridge.
    :param joint: the arm joint the feedback path is traced on.
    :param joint_states_topic: the sim's joint states.
    :param velocity_topic: giskard's streamed joint velocities.
    :param controlled_joints: the joints ``velocity_topic`` carries, in its order.
    :param marker_keys: the objects whose moves are drags of a teleop target.
    """

    def __init__(self, node, world, bridge, joint: str, joint_states_topic: str,
                 velocity_topic: str, controlled_joints: List[str], marker_keys):
        self.world = world
        self.bridge = bridge
        self.joint = joint
        self.connection_name = str(world.get_connection_by_name(joint).name)
        self.dof_id = str(world.get_connection_by_name(joint).raw_dof.id)
        self.controlled_joints = list(controlled_joints)
        self.marker_keys = set(marker_keys)
        self._lock = threading.Lock()
        # the sim's samples of the joint: (sim stamp, value, received here)
        self._sim: Deque[Tuple[float, float, float]] = collections.deque()
        self._sim_all: Deque[Tuple[float, np.ndarray]] = collections.deque()
        self._stages: Dict[str, List[float]] = collections.defaultdict(list)
        self._served: List[float] = []
        self._last_move = 0.0
        self._onset: Optional[Dict[str, float]] = None
        self._arm_index: Optional[List[int]] = None

        group = ReentrantCallbackGroup()
        node.create_subscription(JointState, joint_states_topic, self._on_sim, 50,
                                 callback_group=group)
        node.create_subscription(String, "/semantic_digital_twin/world_sync",
                                 self._on_world_sync, 1000, callback_group=group)
        node.create_subscription(Float64MultiArray, velocity_topic, self._on_velocity,
                                 10, callback_group=group)
        self._wrap_bridge()
        threading.Thread(target=self._report, daemon=True, name="latency-report").start()
        print(f"latency trace: feedback on {joint}, command on drags of "
              f"{', '.join(sorted(self.marker_keys))}; a line every {REPORT_PERIOD:g}s",
              flush=True)

    # %% feedback: the sim's samples, and where each later stage shows them
    def _on_sim(self, message: JointState) -> None:
        now = time.time()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        names = list(message.name)
        with self._lock:
            if self.joint in names:
                value = message.position[names.index(self.joint)]
                self._sim.append((stamp, value, now))
                if self._moving():
                    self._stages["sim -> demo (ros)"].append(now - stamp)
            if self._arm_index is None:
                self._arm_index = [names.index(j) for j in names if "_fr3_joint" in j]
            self._sim_all.append((now, np.asarray(message.position)[self._arm_index]))
            while self._sim and now - self._sim[0][2] > HISTORY:
                self._sim.popleft()
            while self._sim_all and now - self._sim_all[0][0] > HISTORY:
                self._sim_all.popleft()
        self._check_onset_arm(now)

    def _moving(self) -> bool:
        return len(self._sim) >= 2 and abs(self._sim[-1][1] - self._sim[-2][1]) > MOVING

    def _age_of(self, value: float, now: float) -> Optional[float]:
        """How long ago the sim published ``value``, if the joint was moving then."""
        with self._lock:
            samples = list(self._sim)
        for index in range(len(samples) - 1, 0, -1):
            stamp, sample, _ = samples[index]
            if abs(sample - value) <= MATCH:
                if abs(sample - samples[index - 1][1]) <= MOVING:
                    return None          # at rest: any old sample would match
                return now - stamp
        return None

    def _record(self, stage: str, value: Optional[float], now: float) -> None:
        if value is None:
            return
        age = self._age_of(value, now)
        if age is not None:
            with self._lock:
                self._stages[stage].append(age)

    def _on_world_sync(self, message: String) -> None:
        now = time.time()
        if '"giskard"' not in message.data[:400] or self.dof_id not in message.data:
            return
        content = json.loads(message.data)
        if (content.get("meta_data") or {}).get("node_name") != "giskard":
            return
        update = content.get("state_update") or {}
        for index, entry in enumerate(update.get("ids") or []):
            if (entry or {}).get("value") == self.dof_id:
                self._record("sim -> giskard /world_sync", float(update["states"][index]), now)
                return

    def _wrap_bridge(self) -> None:
        bridge = self.bridge
        snapshot, get_state = bridge.snapshot, bridge.get_state
        queue_move, apply_moves = bridge.queue_move, bridge.apply_moves
        connection = self.world.get_connection_by_name(self.joint)

        def traced_snapshot():
            snapshot()
            self._record("sim -> twin + bridge snapshot", float(connection.position), time.time())

        def traced_get_state():
            payload = get_state()
            now = time.time()
            with self._lock:
                self._served.append(now)
            value = (payload.get("frames") or {}).get(self.connection_name)
            self._record("sim -> served to a viewer", value, now)
            return payload

        def traced_queue_move(request):
            now = time.time()
            if request.object_key in self.marker_keys:
                with self._lock:
                    if now - self._last_move > DRAG_GAP and self._onset is None:
                        self._onset = {"http": now}
                    self._last_move = now
            return queue_move(request)

        def traced_apply_moves():
            pending = bool(getattr(bridge, "_moves", None))
            apply_moves()
            if pending:
                with self._lock:
                    if self._onset is not None and "apply" not in self._onset:
                        self._onset["apply"] = time.time()

        bridge.snapshot = traced_snapshot
        bridge.get_state = traced_get_state
        bridge.queue_move = traced_queue_move
        bridge.apply_moves = traced_apply_moves

    # %% command: a drag's first move through to the arm moving
    def _on_velocity(self, message: Float64MultiArray) -> None:
        now = time.time()
        with self._lock:
            onset = self._onset
            if onset is None or "apply" not in onset or "command" in onset:
                return
            arm = [i for i, name in enumerate(self.controlled_joints) if "_fr3_joint" in name]
            if any(abs(message.data[i]) > 1e-3 for i in arm if i < len(message.data)):
                onset["command"] = now

    def _check_onset_arm(self, now: float) -> None:
        with self._lock:
            onset = self._onset
            if onset is None:
                return
            if now - onset["http"] > ONSET_TIMEOUT:
                print(f"[latency] drag onset: no arm motion within {ONSET_TIMEOUT:g}s "
                      f"(stages seen: {', '.join(k for k in onset if k != 'http')})", flush=True)
                self._onset = None
                return
            if "command" not in onset:
                return
            before = [q for t, q in self._sim_all if t <= onset["http"]]
            if not before or not self._sim_all:
                return
            moved = np.max(np.abs(self._sim_all[-1][1] - before[-1])) > 1e-3
            if not moved:
                return
            onset["arm"] = now
            self._onset = None
        stages = [("http", "apply"), ("apply", "command"), ("command", "arm")]
        parts = ", ".join(
            f"{a}->{b} {_ms(onset[b] - onset[a])}" for a, b in stages if a in onset and b in onset
        )
        print(f"[latency] drag onset: {parts} ms; total {_ms(onset['arm'] - onset['http'])} ms",
              flush=True)

    # %% the summary
    def _report(self) -> None:
        while True:
            time.sleep(REPORT_PERIOD)
            now = time.time()
            with self._lock:
                stages, self._stages = self._stages, collections.defaultdict(list)
                served = [t for t in self._served if now - t <= REPORT_PERIOD]
                self._served = []
            gaps = np.diff(served) if len(served) > 1 else []
            print(f"[latency] feedback ({self.joint}, only while it moves):", flush=True)
            for stage in ("sim -> demo (ros)", "sim -> giskard /world_sync",
                          "sim -> twin + bridge snapshot", "sim -> served to a viewer"):
                print(f"[latency]   {stage:32s} {_summary(stages.get(stage, []))}", flush=True)
            print(f"[latency]   viewer reads: {len(served) / REPORT_PERIOD:.1f}/s, "
                  f"longest gap {_ms(max(gaps)) if len(gaps) else '--'} ms", flush=True)

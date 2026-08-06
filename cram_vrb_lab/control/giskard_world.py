"""Putting a robot and a scene into one giskard world, for any pair of the two.

A giskard world config builds the robot; the scenes in this repo are merged onto
``map`` next to it so giskard can avoid the walls and the furniture. That merge is
the same three lines for every robot and every environment, and the only thing
that varies is *what* is merged and *where* -- which is what an
:class:`~cram_vrb_lab.specs.EnvironmentSpec` carries. So instead of one world
config class per robot x scene combination, :func:`build_world_config` composes
the robot's own config with :class:`WithEnvironment` at run time.

Purely additive: no changes to giskardpy or semantic_digital_twin. The robot half
is whatever config the robot already has -- giskardpy's stock
``WorldWithStretchConfigDiffDrive`` for the Stretch, this repo's
:class:`~cram_vrb_lab.robots.panda.giskard_config.WorldWithPandaConfig` for the
Panda -- and the environment is merged after it has run.

.. warning::
   **The merge order is load-bearing.** The environment must be merged *after*
   the robot, which is what :meth:`WithEnvironment.setup_world` does by calling
   ``super().setup_world()`` first. Reason:
   :class:`~cram_vrb_lab.robots.stretch.giskard_config.StretchRealStyleInterface`
   identifies the ``map -> odom`` localization joint as
   ``world.get_connections_by_type(Connection6DoF)[0]`` -- by position, not by
   name -- and an environment can contribute ``Connection6DoF`` connections of its
   own (the garmi-apartment MJCF contributes 15, for its free bodies). ``[0]`` is
   the right one only because the robot's ``setup_world`` creates the localization
   connection before anything else. Merge the environment first and giskard would
   sync a book to the robot's localization transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.world_description.connections import FixedConnection

from cram_vrb_lab.specs import EnvironmentSpec


@dataclass
class WithEnvironment:
    """Mixin: merge an environment onto ``map`` once the robot world is built.

    Only ever used through :func:`build_world_config`, which mixes it in front of
    a concrete world config -- ``super().setup_world()`` below is that config's.
    """

    environment: Optional[EnvironmentSpec] = field(kw_only=True, default=None)
    """The scenery to merge, or ``None`` to plan against the robot alone."""

    def setup_world(self) -> None:
        """Build the robot world as usual, then merge the environment onto ``map``.

        Runs inside the ``modify_world`` context that
        :class:`giskardpy.middleware.ros2.giskard.Giskard` already opens around
        ``setup_world`` (matching the world configs, which likewise assume it).
        """
        super().setup_world()

        if self.environment is None:
            return

        environment_world = self.environment.load()
        self.world.merge_world(
            environment_world,
            FixedConnection(
                parent=self.world.root,
                child=environment_world.root,
                parent_T_connection_expression=self.environment.pose(),
            ),
        )


def build_world_config(
    robot_world_config: type,
    environment: Optional[EnvironmentSpec] = None,
    **kwargs,
):
    """Instantiate ``robot_world_config``, with ``environment`` merged in after it.

    :param robot_world_config: a world config class that builds *only* the robot.
    :param environment: the scenery to add, or ``None`` for the robot alone --
        in which case the class is instantiated unchanged.
    :param kwargs: passed straight to the config (the ``urdf``, ...).

    The composed class is built per call rather than declared per combination;
    that is the point of this module. ``dataclass()`` has to be re-applied because
    the new class has fields from both halves and needs an ``__init__`` that
    accepts them.
    """
    if environment is None:
        return robot_world_config(**kwargs)

    composed = dataclass(
        type(
            f"{robot_world_config.__name__}WithEnvironment",
            (WithEnvironment, robot_world_config),
            {},
        )
    )
    return composed(environment=environment, **kwargs)

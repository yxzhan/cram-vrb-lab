#!/usr/bin/env python3
"""Mark chairs, tables, and books as dynamic free bodies in world.usda.

These assets already ship with PhysicsRigidBodyAPI + PhysicsCollisionAPI +
PhysicsMeshCollisionAPI on their `Geom/body` Mesh — they just weren't moving
because the world.usda override pinned them to `physics:approximation =
"meshSimplification"`, and a triangle-mesh approximation forces PhysX to
treat the body as static/kinematic.

This script:
  - Switches each targeted Actor's `Geom/body` to `approximation = "convexHull"`
    (the cheapest approximation that PhysX accepts for dynamic bodies).
  - Drops the legacy `PhysxTriangleMeshSimplificationCollisionAPI` override.
  - Adds PhysicsMassAPI + an explicit mass to `Geom/body` so behaviour doesn't
    depend on PhysX's density-based default.
  - Does NOT touch the Actor Xform itself (adding RigidBodyAPI there would nest
    rigid bodies and PhysX rejects that).

Idempotent: re-running just normalizes the same state."""

from pxr import Usd, UsdPhysics, Sdf

WORLD = '/mnt/dev-tools/garmi-scene/assets/raw/garmi-apartment/world.usda'
SCOPE_ROOT = '/Root/Meshes/Assets'

# scope name -> mass in kg for each Actor in that scope
TARGETS = {
    'chair':       5.0,
    'CoffeeTable': 15.0,
    'DiningTable': 25.0,
    'book':        0.5,
}

# Schemas that need to be stripped from previous attempts.
ACTOR_APIS_TO_DROP = ('PhysicsRigidBodyAPI', 'PhysicsMassAPI')
BODY_APIS_TO_DROP  = ('PhysxTriangleMeshSimplificationCollisionAPI',)


def strip_apis(prim_spec, names):
    """Surgically remove entries from a prim spec's apiSchemas list-op.
    Returns True if anything changed."""
    if not prim_spec or 'apiSchemas' not in prim_spec.ListInfoKeys():
        return False
    op = prim_spec.GetInfo('apiSchemas')
    new_prepended = tuple(x for x in op.prependedItems if x not in names)
    new_appended  = tuple(x for x in op.appendedItems  if x not in names)
    new_explicit  = tuple(x for x in op.explicitItems  if x not in names)
    if (new_prepended == op.prependedItems
            and new_appended == op.appendedItems
            and new_explicit == op.explicitItems):
        return False
    if op.isExplicit:
        new_op = Sdf.TokenListOp.CreateExplicit(new_explicit)
    else:
        new_op = Sdf.TokenListOp.Create(
            prependedItems=new_prepended,
            appendedItems=new_appended,
            deletedItems=tuple(op.deletedItems),
        )
    if not (new_prepended or new_appended or new_explicit
            or op.deletedItems):
        prim_spec.ClearInfo('apiSchemas')
    else:
        prim_spec.SetInfo('apiSchemas', new_op)
    return True


def main():
    stage = Usd.Stage.Open(WORLD)
    if not stage:
        raise SystemExit(f'cannot open {WORLD}')
    layer = stage.GetRootLayer()

    total = 0
    for scope_name, mass in TARGETS.items():
        scope = stage.GetPrimAtPath(f'{SCOPE_ROOT}/{scope_name}')
        if not scope or not scope.IsValid():
            print(f'WARN: scope {scope_name} not found, skipping')
            continue

        for actor in scope.GetChildren():
            if not actor.GetName().startswith('Actor_'):
                continue

            # Strip the bogus Actor-level RigidBody/Mass we might have added.
            actor_spec = layer.GetPrimAtPath(actor.GetPath())
            strip_apis(actor_spec, ACTOR_APIS_TO_DROP)
            if actor_spec is not None and 'physics:mass' in actor_spec.attributes:
                actor_spec.RemoveProperty(actor_spec.attributes['physics:mass'])

            # Body Mesh: drop legacy simplification API and set convexHull + mass.
            body = stage.OverridePrim(f'{actor.GetPath()}/Geom/body')
            body_spec = layer.GetPrimAtPath(body.GetPath())
            strip_apis(body_spec, BODY_APIS_TO_DROP)

            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(body)
            mesh_col.CreateApproximationAttr().Set('convexHull')
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(float(mass))
            # The source assets ship with physics:kinematicEnabled = true (so
            # they sit as decor by default). Override to false so the body
            # actually falls under gravity / responds to pushes.
            UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr().Set(False)

            total += 1
            print(f'  {scope_name:12s}  {actor.GetName():12s}  mass={mass:.1f}kg')

    layer.Save()
    print(f'updated {total} actors -> wrote {WORLD}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate cabinet_XXX_articulated.usda: each door / drawer / handle becomes
its own Xform-wrapped link with a friendly name, and the geometry is copied
verbatim from the original asset's Mesh into a `geom` child of that wrapper.

We don't sublayer the original cabinet asset (only its `_look.usda` for
materials) because we want to fully control the prim layout — using `over`
on Mesh prims to attach RigidBodyAPI proved unreliable for PhysX articulations
in Isaac Sim ("cannot create a joint between static bodies"). Wrapping each
Mesh in an Xform with the physics APIs matches the pattern used for doors,
which works."""

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf, Vt
import numpy as np
import os

ASSET_DIR = '/mnt/dev-tools/garmi-scene/assets/raw/garmi-apartment/Assets/cabinet_A71cf8c99Bd8a4ab0847b101e3f6f9fa4'
NAME = 'cabinet_A71cf8c99Bd8a4ab0847b101e3f6f9fa4'
SRC = f'{ASSET_DIR}/{NAME}.usda'
OUT = f'{ASSET_DIR}/{NAME}_articulated.usda'

DRAWER_Y_THRESHOLD = 0.1            # y-thickness above this = drawer, else door
PANEL_MASS = 5.0                    # kg, doors
DRAWER_MASS = 8.0                   # kg, drawers
HANDLE_MASS = 0.2                   # kg, handles
DOOR_LIMIT_DEG = 90.0               # max outward swing
DRAWER_TRAVEL_FRACTION = 0.9


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def mesh_bbox(prim):
    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def copy_mesh(stage, dest_path, src_mesh_prim):
    """Copy a Mesh prim's geometry + GeomSubsets + material bindings to a new
    location. The destination Mesh is identity-transformed (the caller sets up
    the transform on the wrapper Xform); per-vertex/face arrays and subdivision
    attrs are preserved."""
    src = UsdGeom.Mesh(src_mesh_prim)
    dst = UsdGeom.Mesh.Define(stage, dest_path)

    dst.CreatePointsAttr().Set(src.GetPointsAttr().Get())
    dst.CreateFaceVertexCountsAttr().Set(src.GetFaceVertexCountsAttr().Get())
    dst.CreateFaceVertexIndicesAttr().Set(src.GetFaceVertexIndicesAttr().Get())

    normals = src.GetNormalsAttr().Get()
    if normals is not None:
        dst.CreateNormalsAttr().Set(normals)
        interp = src.GetNormalsInterpolation()
        if interp:
            dst.SetNormalsInterpolation(interp)

    # primvars (st, displayColor, etc.)
    src_api = UsdGeom.PrimvarsAPI(src_mesh_prim)
    dst_api = UsdGeom.PrimvarsAPI(dst)
    for pv in src_api.GetPrimvars():
        if pv.HasAuthoredValue() or pv.IsIndexed():
            new_pv = dst_api.CreatePrimvar(
                pv.GetPrimvarName(), pv.GetTypeName(), pv.GetInterpolation()
            )
            v = pv.Get()
            if v is not None:
                new_pv.Set(v)

    # subdivision-related attrs (preserve catmullClark / interp boundary etc.)
    for getter, setter in [
        ('GetSubdivisionSchemeAttr', 'CreateSubdivisionSchemeAttr'),
        ('GetInterpolateBoundaryAttr', 'CreateInterpolateBoundaryAttr'),
        ('GetTriangleSubdivisionRuleAttr', 'CreateTriangleSubdivisionRuleAttr'),
        ('GetFaceVaryingLinearInterpolationAttr', 'CreateFaceVaryingLinearInterpolationAttr'),
        ('GetDoubleSidedAttr', 'CreateDoubleSidedAttr'),
        ('GetOrientationAttr', 'CreateOrientationAttr'),
    ]:
        v = getattr(src, getter)().Get()
        if v is not None and v != '':
            getattr(dst, setter)().Set(v)

    # GeomSubsets (material bindings on faces)
    for c in src_mesh_prim.GetChildren():
        if c.GetTypeName() != 'GeomSubset':
            continue
        sub = UsdGeom.Subset.Define(stage, f'{dest_path}/{c.GetName()}')
        src_sub = UsdGeom.Subset(c)
        sub.CreateFamilyNameAttr().Set(src_sub.GetFamilyNameAttr().Get() or 'materialBind')
        sub.CreateElementTypeAttr().Set(src_sub.GetElementTypeAttr().Get() or 'face')
        sub.CreateIndicesAttr().Set(src_sub.GetIndicesAttr().Get())
        mb = UsdShade.MaterialBindingAPI(c).GetDirectBindingRel()
        targets = mb.GetTargets() if mb else []
        if targets:
            rel = sub.GetPrim().CreateRelationship('material:binding', custom=False)
            rel.SetTargets(targets)

    return dst


def set_translate(xform_prim, t):
    """Add a single xformOp:translate to an Xform prim."""
    x = UsdGeom.Xformable(xform_prim)
    op = x.AddTranslateOp()
    op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))


def classify_and_compute(panel_prim, handle_prim):
    """Return panel center/size, classification (door/drawer), joint anchor,
    axis, limits, mass. Position values are in raw mesh-local coords (== the
    cabinet-local frame, since every panel shares the same translate)."""
    p_min, p_max = mesh_bbox(panel_prim)
    pc = (p_min + p_max) / 2.0
    ps = p_max - p_min
    h_min, h_max = mesh_bbox(handle_prim)
    hc = (h_min + h_max) / 2.0
    dx = hc[0] - pc[0]

    if ps[1] > DRAWER_Y_THRESHOLD:
        travel = ps[1] * DRAWER_TRAVEL_FRACTION
        return {
            'type': 'drawer',
            'panel_center': pc,
            'panel_size': ps,
            'handle_center': hc,
            'axis': 'Y',
            'anchor': pc,
            'lower_limit': -travel,
            'upper_limit': 0.0,
            'mass': DRAWER_MASS,
        }
    if dx < -0.02:
        hinge_x = p_max[0]
        hinge_is_right = True
    elif dx > 0.02:
        hinge_x = p_min[0]
        hinge_is_right = False
    else:
        hinge_x = p_max[0]
        hinge_is_right = True

    # The cabinet "front" is the side the handle bulges out from. Handle y is
    # offset from panel y in the outward direction.
    front_y_sign = -1.0 if hc[1] < pc[1] else 1.0
    # With axis=Z, a right-hinge door at θ=+90 swings its handle in the -Y
    # direction (and a left-hinge door swings it in +Y). So "outward" corresponds
    # to +angle iff -front_y_sign (right hinge) / +front_y_sign (left hinge).
    if hinge_is_right:
        outward_sign = -front_y_sign
    else:
        outward_sign = front_y_sign
    if outward_sign > 0:
        lower_limit, upper_limit = 0.0, DOOR_LIMIT_DEG
    else:
        lower_limit, upper_limit = -DOOR_LIMIT_DEG, 0.0

    return {
        'type': 'door',
        'panel_center': pc,
        'panel_size': ps,
        'handle_center': hc,
        'axis': 'Z',
        'anchor': np.array([hinge_x, pc[1], pc[2]]),
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'mass': PANEL_MASS,
    }


# ----------------------------------------------------------------------------
# main authoring
# ----------------------------------------------------------------------------

def main():
    src_stage = Usd.Stage.Open(SRC, Usd.Stage.LoadAll)
    geom = src_stage.GetPrimAtPath(f'/{NAME}/Geom')
    static_prim = src_stage.GetPrimAtPath(f'/{NAME}/Geom/Static')
    assert geom and static_prim, 'cabinet asset is missing /Geom or /Geom/Static'

    # All panels share the same xformOp:translate — read it once.
    static_t = static_prim.GetAttribute('xformOp:translate').Get()
    cabinet_translate = (float(static_t[0]), float(static_t[1]), float(static_t[2]))

    # Bucket children: skip Static, pair Mesh + Mesh_handle_1
    pairs = {}  # base_name -> {'panel': prim, 'handle': prim}
    for c in geom.GetChildren():
        if c.GetTypeName() != 'Mesh' or c.GetName() == 'Static':
            continue
        n = c.GetName()
        if n.endswith('_handle_1'):
            base = n[: -len('_handle_1')]
            pairs.setdefault(base, {})['handle'] = c
        else:
            pairs.setdefault(n, {})['panel'] = c

    # Classify everything.
    panels = []
    for base, parts in pairs.items():
        info = classify_and_compute(parts['panel'], parts['handle'])
        info['orig_base'] = base
        info['panel_prim'] = parts['panel']
        info['handle_prim'] = parts['handle']
        panels.append(info)

    # Friendly-name assignment: doors and drawers separately, sorted by
    # (x ascending, z descending) so "1" is the top-left of each group.
    doors = sorted(
        [p for p in panels if p['type'] == 'door'],
        key=lambda p: (round(p['panel_center'][0], 3), -p['panel_center'][2]),
    )
    drawers = sorted(
        [p for p in panels if p['type'] == 'drawer'],
        key=lambda p: (round(p['panel_center'][0], 3), -p['panel_center'][2]),
    )
    for i, d in enumerate(doors, 1):
        d['link_name'] = f'door_{i}'
    for i, d in enumerate(drawers, 1):
        d['link_name'] = f'drawer_{i}'

    # ---- author articulated layer -----------------------------------------
    if os.path.exists(OUT):
        os.remove(OUT)
    out = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)

    # Pull in Materials via the look sublayer; nothing else from the original
    # asset is referenced — we copy geometry into our wrapper structure below.
    out.GetRootLayer().subLayerPaths.append(f'./{NAME}_look.usda')

    root_path = f'/{NAME}'
    root = out.DefinePrim(root_path, 'Xform')
    out.SetDefaultPrim(root)
    # Intentionally no ArticulationRootAPI: PhysX disallows kinematic bodies
    # inside an articulation, and Static (the cabinet enclosure) needs to stay
    # kinematic so it can use a triangle-mesh collider for its concave interior.
    # Each joint is treated as an independent maximal-coordinate constraint,
    # which is fine for a cabinet (all joints are parallel from the static base).

    # --- Static (kinematic rigid body, full triangle-mesh collider) --------
    static_wrap = UsdGeom.Xform.Define(out, f'{root_path}/Static').GetPrim()
    set_translate(static_wrap, cabinet_translate)
    UsdPhysics.RigidBodyAPI.Apply(static_wrap).CreateKinematicEnabledAttr(True)
    static_mesh = copy_mesh(out, f'{root_path}/Static/geom', static_prim)
    UsdPhysics.CollisionAPI.Apply(static_mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(static_mesh.GetPrim()).CreateApproximationAttr().Set('none')

    # --- Per-link wrappers (door/drawer + handle, both as dynamic bodies) --
    for info in panels:
        link_name = info['link_name']
        link_path = f'{root_path}/{link_name}'
        handle_path = f'{root_path}/{link_name}_handle'

        # panel link
        link_xf = UsdGeom.Xform.Define(out, link_path).GetPrim()
        set_translate(link_xf, cabinet_translate)
        UsdPhysics.RigidBodyAPI.Apply(link_xf)
        UsdPhysics.MassAPI.Apply(link_xf).CreateMassAttr().Set(float(info['mass']))
        link_geom = copy_mesh(out, f'{link_path}/geom', info['panel_prim'])
        UsdPhysics.CollisionAPI.Apply(link_geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(link_geom.GetPrim()).CreateApproximationAttr().Set('convexHull')

        # handle link
        handle_xf = UsdGeom.Xform.Define(out, handle_path).GetPrim()
        set_translate(handle_xf, cabinet_translate)
        UsdPhysics.RigidBodyAPI.Apply(handle_xf)
        UsdPhysics.MassAPI.Apply(handle_xf).CreateMassAttr().Set(HANDLE_MASS)
        handle_geom = copy_mesh(out, f'{handle_path}/geom', info['handle_prim'])
        UsdPhysics.CollisionAPI.Apply(handle_geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(handle_geom.GetPrim()).CreateApproximationAttr().Set('convexHull')

    # --- joints ------------------------------------------------------------
    joints_path = f'{root_path}/joints'
    out.DefinePrim(joints_path, 'Scope')
    static_path = f'{root_path}/Static'

    # All links share the same xformOp:translate, so a point at (X, Y, Z) in
    # raw mesh coords corresponds to the same physical point in every link's
    # local frame — joint localPos0/1 can both use the raw coords directly.
    for info in panels:
        link_path = f'{root_path}/{info["link_name"]}'
        handle_path = f'{link_path}_handle'
        anchor = info['anchor']
        if info['type'] == 'door':
            joint = UsdPhysics.RevoluteJoint.Define(out, f'{joints_path}/{info["link_name"]}_hinge')
            joint.CreateAxisAttr().Set('Z')
        else:
            joint = UsdPhysics.PrismaticJoint.Define(out, f'{joints_path}/{info["link_name"]}_slide')
            joint.CreateAxisAttr().Set('Y')
        joint.CreateBody0Rel().SetTargets([Sdf.Path(static_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(anchor[0]), float(anchor[1]), float(anchor[2])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(anchor[0]), float(anchor[1]), float(anchor[2])))
        joint.CreateLowerLimitAttr().Set(float(info['lower_limit']))
        joint.CreateUpperLimitAttr().Set(float(info['upper_limit']))

        fj = UsdPhysics.FixedJoint.Define(out, f'{joints_path}/{info["link_name"]}_handle_fixed')
        fj.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
        fj.CreateBody1Rel().SetTargets([Sdf.Path(handle_path)])
        h = info['handle_center']
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(h[0]), float(h[1]), float(h[2])))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(float(h[0]), float(h[1]), float(h[2])))

    out.GetRootLayer().Save()
    print(f'wrote {OUT}')
    print(f'  doors:   {len(doors)}   drawers: {len(drawers)}')
    print('  name mapping (link -> original mesh):')
    for info in panels:
        pc = info['panel_center']
        print(
            f'    {info["link_name"]:10s} ({info["type"]:6s}) '
            f'center=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:+.3f}) '
            f'axis={info["axis"]} limits=[{info["lower_limit"]:+.2f}, {info["upper_limit"]:+.2f}] '
            f'orig={info["orig_base"]}'
        )


if __name__ == '__main__':
    main()

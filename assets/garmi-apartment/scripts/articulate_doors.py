#!/usr/bin/env python3
"""Split each door asset's body Mesh into a static frame + dynamic leaf
connected by a PhysicsRevoluteJoint, written to door_*_articulated.usda."""

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf, Vt
import numpy as np
import os
import sys

ROOT = '/mnt/dev-tools/garmi-scene/assets/raw/garmi-apartment/Assets'

# Hinge x position in door-local coords. Determined by handle-opposite-side
# rule from per-subset bbox inspection:
#   door1: handle on left (x≈-0.5) → hinge on right (x=+0.69)
#   door2: handle on right (x≈+0.5) → hinge on left (x=-0.69)
#   door3: handle on right (x≈+0.29) → hinge on left (x=-0.49)
DOORS = [
    # Each door opens 90° inward only. If a particular door swings the wrong
    # way, flip its (lower, upper) from (0, 90) to (-90, 0) (or vice versa).
    {'name': 'door_YWL53QRVALHRQPTUKM888888', 'hinge_x': +0.690, 'lower': 0.0, 'upper': 90.0},
    {'name': 'door_YWL5AMBVAJV2EPTULY888888', 'hinge_x': -0.690, 'lower': 0.0, 'upper': 90.0},
    {'name': 'door_YWL5ERJVALF64PTULM888888', 'hinge_x': -0.490, 'lower': 0.0, 'upper': 90.0},
]

FRAME_SUBSETS = {'mesh_0000'}  # outer doorframe; everything else swings with leaf


def split_mesh(src_mesh_prim, keep_subset_names):
    src = UsdGeom.Mesh(src_mesh_prim)
    points = np.array(src.GetPointsAttr().Get(), dtype=np.float32)
    fvc = np.array(src.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    fvi = np.array(src.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    normals = np.array(src.GetNormalsAttr().Get(), dtype=np.float32)
    st_pv = UsdGeom.PrimvarsAPI(src_mesh_prim).GetPrimvar('st')
    st = np.array(st_pv.Get(), dtype=np.float32)

    offsets = np.concatenate([[0], np.cumsum(fvc)]).astype(np.int64)

    face_mask = np.zeros(len(fvc), dtype=bool)
    subset_info = []
    for child in src_mesh_prim.GetChildren():
        if child.GetTypeName() != 'GeomSubset':
            continue
        if child.GetName() not in keep_subset_names:
            continue
        idx = np.array(UsdGeom.Subset(child).GetIndicesAttr().Get(), dtype=np.int64)
        face_mask[idx] = True
        mat_targets = UsdShade.MaterialBindingAPI(child).GetDirectBindingRel().GetTargets()
        subset_info.append({
            'name': child.GetName(),
            'old_faces': idx,
            'material': str(mat_targets[0]) if mat_targets else None,
        })

    selected_faces = np.where(face_mask)[0]
    old_to_new_face = -np.ones(len(fvc), dtype=np.int64)
    old_to_new_face[selected_faces] = np.arange(len(selected_faces))

    # Build new face-varying arrays + collect used vertices in order of first sight.
    used_old_verts = []
    seen = {}
    new_fvc = []
    new_fvi = []
    new_normals = []
    new_st = []
    for old_fi in selected_faces:
        start = offsets[old_fi]
        end = offsets[old_fi + 1]
        new_fvc.append(int(end - start))
        for k in range(start, end):
            old_v = int(fvi[k])
            if old_v not in seen:
                seen[old_v] = len(used_old_verts)
                used_old_verts.append(old_v)
            new_fvi.append(seen[old_v])
            new_normals.append(normals[k])
            new_st.append(st[k])

    new_points = points[used_old_verts]

    for s in subset_info:
        s['new_faces'] = old_to_new_face[s['old_faces']].astype(np.int32)

    return {
        'points': new_points,
        'face_vertex_counts': np.array(new_fvc, dtype=np.int32),
        'face_vertex_indices': np.array(new_fvi, dtype=np.int32),
        'normals': np.array(new_normals, dtype=np.float32),
        'st': np.array(new_st, dtype=np.float32),
        'subsets': subset_info,
    }


def author_mesh(stage, path, data, copy_subdiv_from_prim):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(data['points']))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(data['face_vertex_counts']))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(data['face_vertex_indices']))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(data['normals']))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    st_pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st_pv.Set(Vt.Vec2fArray.FromNumpy(data['st']))

    src = UsdGeom.Mesh(copy_subdiv_from_prim)
    mesh.CreateSubdivisionSchemeAttr().Set(src.GetSubdivisionSchemeAttr().Get())
    mesh.CreateInterpolateBoundaryAttr().Set(src.GetInterpolateBoundaryAttr().Get())
    mesh.CreateTriangleSubdivisionRuleAttr().Set(src.GetTriangleSubdivisionRuleAttr().Get())
    mesh.CreateFaceVaryingLinearInterpolationAttr().Set(
        src.GetFaceVaryingLinearInterpolationAttr().Get()
    )
    mesh.CreateDoubleSidedAttr().Set(src.GetDoubleSidedAttr().Get())
    mesh.CreateOrientationAttr().Set(src.GetOrientationAttr().Get())

    for s in data['subsets']:
        sub = UsdGeom.Subset.Define(stage, f'{path}/{s["name"]}')
        sub.CreateFamilyNameAttr().Set('materialBind')
        sub.CreateElementTypeAttr().Set('face')
        sub.CreateIndicesAttr().Set(Vt.IntArray.FromNumpy(s['new_faces']))
        if s['material']:
            mat_prim = stage.GetPrimAtPath(s['material'])
            if mat_prim and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI(sub.GetPrim()).Bind(UsdShade.Material(mat_prim))
            else:
                # bind by relationship target even if prim isn't resolvable on this stage yet
                rel = sub.GetPrim().CreateRelationship('material:binding', custom=False)
                rel.SetTargets([Sdf.Path(s['material'])])
    return mesh


def articulate_door(door):
    name = door['name']
    hinge_x = door['hinge_x']
    src_path = f'{ROOT}/{name}/{name}.usda'
    out_path = f'{ROOT}/{name}/{name}_articulated.usda'

    src_stage = Usd.Stage.Open(src_path, Usd.Stage.LoadAll)
    body_prim = src_stage.GetPrimAtPath(f'/{name}/Geom/body')
    assert body_prim and body_prim.IsValid(), f'no body Mesh in {src_path}'

    all_subsets = {
        c.GetName() for c in body_prim.GetChildren() if c.GetTypeName() == 'GeomSubset'
    }
    leaf_subsets = all_subsets - FRAME_SUBSETS

    frame_data = split_mesh(body_prim, FRAME_SUBSETS)
    leaf_data = split_mesh(body_prim, leaf_subsets)

    if os.path.exists(out_path):
        os.remove(out_path)
    out_stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)

    # Sublayer the original look file so /door_XXX/Materials is available
    out_stage.GetRootLayer().subLayerPaths.append(f'./{name}_look.usda')

    root = out_stage.DefinePrim(f'/{name}', 'Xform')
    out_stage.SetDefaultPrim(root)

    # --- Frame (kinematic body so we can be a joint endpoint) -----------------
    frame_path = f'/{name}/Frame'
    frame = UsdGeom.Xform.Define(out_stage, frame_path)
    UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
    UsdPhysics.RigidBodyAPI(frame.GetPrim()).CreateKinematicEnabledAttr(True)

    frame_body_path = f'{frame_path}/body'
    fm = author_mesh(out_stage, frame_body_path, frame_data, body_prim)
    UsdPhysics.CollisionAPI.Apply(fm.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(fm.GetPrim()).CreateApproximationAttr().Set('none')

    # --- Leaf (dynamic) -------------------------------------------------------
    leaf_path = f'/{name}/Leaf'
    leaf = UsdGeom.Xform.Define(out_stage, leaf_path)
    UsdPhysics.RigidBodyAPI.Apply(leaf.GetPrim())
    UsdPhysics.MassAPI.Apply(leaf.GetPrim()).CreateMassAttr().Set(15.0)

    leaf_body_path = f'{leaf_path}/body'
    lm = author_mesh(out_stage, leaf_body_path, leaf_data, body_prim)
    UsdPhysics.CollisionAPI.Apply(lm.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(lm.GetPrim()).CreateApproximationAttr().Set('convexHull')

    # --- Revolute joint -------------------------------------------------------
    joint = UsdPhysics.RevoluteJoint.Define(out_stage, f'/{name}/HingeJoint')
    joint.CreateBody0Rel().SetTargets([Sdf.Path(frame_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(leaf_path)])
    joint.CreateAxisAttr().Set('Z')
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(hinge_x, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(hinge_x, 0.0, 0.0))
    joint.CreateLowerLimitAttr().Set(float(door['lower']))
    joint.CreateUpperLimitAttr().Set(float(door['upper']))

    out_stage.GetRootLayer().Save()
    print(f'  wrote {out_path}')
    print(f'    frame: {len(frame_data["points"])} verts, '
          f'{len(frame_data["face_vertex_counts"])} faces')
    print(f'    leaf:  {len(leaf_data["points"])} verts, '
          f'{len(leaf_data["face_vertex_counts"])} faces, hinge_x={hinge_x:+.3f}')


def main():
    for d in DOORS:
        print(f'[{d["name"]}]')
        articulate_door(d)


if __name__ == '__main__':
    main()

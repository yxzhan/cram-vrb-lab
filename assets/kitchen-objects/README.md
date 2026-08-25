# Kitchen objects

Four graspable kitchen props, each a standalone USD asset extracted from
`assets/apartment/apartmentICRA.usda`. The source scene is unchanged -- these are
copies, so the apartment still renders exactly as before.

Each directory is self-contained -- no asset here reads a single file from
outside its own folder. The layout mirrors the source scene
(`meshes/usd/`, `materials/usd/`, `textures/`), which is what keeps the
`../../textures/...` paths inside the material files valid without rewriting
them: from `<asset>/materials/usd/` that resolves to `<asset>/textures/`, so it
stays within the asset despite the `../..` reading like an escape.

    SM_Cup/
      SM_Cup.usda          <- entry point, defaultPrim = SM_Cup
      meshes/usd/*.usda    <- visual mesh (also the collision source)
      materials/usd/*.usda
      textures/*.png       <- only for the two textured objects

Reference one from a scene with `@.../SM_Cup/SM_Cup.usda@` and no sub-path; the
layer's `defaultPrim` resolves to the object's root Xform.

## What each asset carries

The root prim of every asset sits at the **origin** with an identity transform,
because an asset that is reusable cannot hard-code where it stood in one
apartment. Where each object *did* stand in that scene is recorded in the `doc`
string of its own `.usda` and repeated here, so the original arrangement on the
kitchen counter can be rebuilt:

| Asset | Apartment pose (world, m) | Mass | Collision | Texture |
|---|---|---|---|---|
| `SM_Cup` | (7.05, -5.17, 0.90433) | 0.120 kg | convex decomposition, <=32 hulls | none (white PBR) |
| `SM_CerealBox` | (6.85, -5.47, 1.0117) | 0.420 kg | 1 convex hull | `M_CerealBox.png` |
| `SM_MilkBox` | (6.85, -4.95, 0.9617) | 1.140 kg | 1 `Cube` primitive | `..._M_MilkBox_BaseColor.png` |
| `SM_SmallBowl` | (7.05, -4.92, 0.89484) | 0.058 kg | convex decomposition, <=32 hulls | none (white PBR) |

`PhysicsRigidBodyAPI` / `PhysicsMassAPI` and the authored inertia tensors came
across with the prims, so each asset is a complete physics body rather than
visual geometry that a caller has to re-mass.

## Where the numbers came from

Everything except `SM_CerealBox`'s mass is the source scene's own value, carried
over verbatim.

**`SM_CerealBox` was re-massed from 4.2 kg to 0.42 kg.** Its collider measures
0.07 x 0.20 x 0.30 m, i.e. 4.2 L, so the original 4.2 kg put the box at exactly
1000 kg/m^3 -- water. That is the signature of a density left at a default rather
than a box that was ever weighed. 0.42 kg gives 100 kg/m^3, which is where
packaged cereal actually sits. `physics:diagonalInertia` was scaled by the same
factor of 10: inertia is linear in mass for unchanged geometry, and leaving it
alone would have left the box resisting rotation like something ten times its
weight. The scaled tensor is still exactly the uniform-box tensor for its
dimensions, which is what the source authored and what a re-derivation confirms.

**`SM_Cup` and `SM_SmallBowl` had their colliders replaced.** The source scene
baked them as 263 and 226 separate `UCX_*` convex hulls. Each hull was tiny
(8.4 vertices on average), so the cost was never geometry but *shape count*:
PhysX generates contacts per shape, and a few hundred shapes on one rigid body is
a lot to carry for a prop that mostly gets grasped and set down.

Both now apply `PhysicsCollisionAPI` directly to their **visual mesh prim** --
one collision source instead of hundreds of sibling prims -- with

    uniform token physics:approximation = "convexDecomposition"
    int physxConvexDecompositionCollision:maxConvexHulls = 32

so PhysX decomposes the mesh at load and caches the result. `convexDecomposition`
rather than the cheaper `convexHull` because both shapes are **hollow** and that
hollowness is the point: a single hull fills in the cup's handle opening and
turns the bowl into a solid dome, which would make it impossible to put anything
in the bowl or hook a finger through the handle. 32 is PhysX's own default and an
8x cut in shape count; raise it if a grasp needs finer contact, lower it if
contact generation shows up in a profile.

The trade is a few seconds of cooking on first load, and that the 489 baked hull
files are gone -- they remain in `assets/apartment/apartmentICRA/meshes/usd/` if
the exact original decomposition is ever wanted back.

Contrast with `cram_vrb_lab/scenes/props/constants.py`, where the pick-and-place
cube is built from numbers on both the Isaac and the giskard side precisely so
the two agree by construction. These assets only describe the Isaac side; a twin
counterpart would still have to be modelled to plan grasps against them.

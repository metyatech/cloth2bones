# Cloth2Bones

Cloth2Bones converts a baked cloth geometry cache into a game-ready skinned
animation, then provides a research baseline for learning cloth helper-bone
motion from body/collider motion.

The supported first workflow is:

```text
rest FBX + animated Alembic
        │
        ├─ Dem Bones decomposition (weights and rigid-bone solve)
        └─ Blender clean-rig rebuild (stable bind/rest space)
             └─ FBX with bones, weights, and animation
```

The Blender clean-rig route is intentional. The Dem Bones solver is useful for
the decomposition, but its standard FBX writer can place bind and animation
transforms in incompatible spaces for some Blender-authored FBX/Alembic input.
The clean route reconstructs the skinning transforms in one Blender scene and
exports them with Blender's own FBX writer.

This project is an OSS PoC, not a promise of arbitrary-pose real-time cloth.
It converts baked simulation motion. The Body Motion PoC learns a compact
body-driven helper-rig baseline from collider motion; it does not yet replace
VRChat PhysBones or provide a universal humanoid cloth solver.

## Requirements

- Windows PowerShell 7 or Windows PowerShell 5.1
- Blender 5.2 or newer (the published PoC was verified with Blender 5.2)
- A Dem Bones executable built or downloaded from the official project
- Python 3.10+ and NumPy for the body-motion analysis tools
- Optional: Unity 2022 LTS or newer for an engine-side import check

Dem Bones is an external BSD-3-Clause dependency ([official repository and
license](https://github.com/electronicarts/dem-bones)). Do not copy its
executable or repository into this project unless its license and
redistribution terms are being followed. The ClothTransformer dataset is also
external ([official dataset card](https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset))
and is released under CC BY 4.0 with additional upstream asset terms; this
repository contains no dataset, mesh, cache, or binary assets.

## Install development dependencies

```powershell
python -m pip install -e . -r requirements-dev.txt
npm install
Install-Module PSScriptAnalyzer -Scope CurrentUser
Install-Module Pester -Scope CurrentUser -Force
```

## Convert a cloth cache

The command takes explicit paths, so it has no machine-specific defaults:

```powershell
pwsh ./tools/run_cloth2bones.ps1 `
  -DemBonesExe 'C:/tools/DemBones.exe' `
  -BlenderExe 'C:/Program Files/Blender Foundation/Blender/blender.exe' `
  -InputFbx ./input/rest.fbx `
  -InputAbc ./input/animation.abc `
  -OutputRoot ./out `
  -BoneCount 50
```

The default export scales match the verified PoC (`0.01` FBX global scale and
`100` clean-armature content scale). For a source that is already in the
desired FBX unit convention, pass `-ExportGlobalScale 1 -ExportContentScale 1`
and compare the resulting cache coordinates before using the FBX in production.

Outputs include the raw Dem Bones FBX, clean skinned FBX, fit report, Blender
acceptance report, and sampled Alembic comparison report. The acceptance gate
requires the requested bone count, no zero-weight vertices, finite evaluated
vertices, and sampled RMS below `0.05` source units by default.

## Body Motion PoC

A ClothTransformer-style NPZ is inspected with:

```powershell
python ./tools/inspect_npz.py ./input/sim_00000.npz --json ./out/npz_report.json
```

First export the clean FBX's weights and bone transforms with Blender:

```powershell
blender --background --factory-startup `
  --python ./tools/dump_blender_rig_poses.py -- `
  --fbx ./out/cloth_clean_rigid_50.fbx `
  --out ./out/cloth_rig_poses.npz
```

Then fit the body-driven baseline. It uses only collider vertices, not frame
numbers, and reports train/validation RMS and maximum point error. The tool
infers the signed axis permutation between the NPZ rest mesh and the clean FBX
from same-index rest vertices:

```powershell
python ./tools/body_motion_poc.py `
  --npz ./input/sim_00000.npz `
  --poses ./out/cloth_rig_poses.npz `
  --out ./out/body_driven_poses.npz `
  --report ./out/body_motion_report.json
```

For a Blender preview FBX:

```powershell
blender --background --factory-startup `
  --python ./tools/apply_body_driven_poses.py -- `
  --fbx ./out/cloth_clean_rigid_50.fbx `
  --poses ./out/body_driven_poses.npz `
  --out ./out/cloth_body_driven.fbx
```

The default body driver is the deterministic full-collider Kabsch transform
applied to the fixed cloth-bone hierarchy. This captures global body motion
without frame lookup and is the verified baseline for this sample. An optional
`--model linear` experiment fits an intercept-plus-ridge mapping from the
region features to each bone transform; its held-out score must be compared
against the global baseline before being adopted.

For the external 240-frame Human Garment sample used during development, the
NPZ contained `traj`, `traj_vel`, `collision_vertices`, `collision_vel`,
`triangles`, `edges`, `collision_triangles`, and `collision_edges`. The body
collider had 560 vertices and 1,116 triangles. With frames 1-180 for training
and 181-240 held out, the rest-pose baseline was RMS `2.5734` and maximum
point error `2.8409`; the global-Kabsch body driver reached RMS `0.03893` and
maximum point error `0.1465` on the held-out frames. These numbers are a
reproducibility reference only; the dataset is not included in this repository.

## Experimental PoC 3.1: body-local drivers

The experimental body-local tool removes the frame-wise rigid transform of the
whole collider before predicting cloth-bone transforms. It uses automatically
discovered residual-motion regions and compares static rest, linear/ridge,
polynomial, nearest-pose, RBF, and region-local models. It never supplies a
frame number, time, animation progress, cloth vertex, or cloth teacher as a
feature. It also evaluates contiguous, interleaved, and motion-space holdouts
and benchmarks 50/32/20/16/8 bone reductions.

Run it with explicit paths:

```powershell
pwsh ./tools/run_body_local_poc.ps1 `
  -Npz ./input/sim_00000.npz `
  -CleanFbx ./out/cloth_clean_rigid_50.fbx `
  -BlenderExe 'C:/Program Files/Blender Foundation/Blender/blender.exe' `
  -OutputRoot ./out/body_local_poc
```

The wrapper exports pose/weight data from the exact clean FBX passed to
`-CleanFbx`, writes the canonicalized report and predicted FBX, checks the FBX
against the canonical teacher, and renders comparison frames. A pose NPZ
from another clean-rig export is rejected when its rest vertices or weights do
not match the target FBX. See [`docs/body_local_poc.md`](docs/body_local_poc.md)
for the protocol and interpretation.

This is an experimental offline baseline, not arbitrary-pose humanoid cloth
inference and not a finished VRChat runtime rig. The current dataset contains
no semantic skeleton, so its collider regions are not yet stable Hips/Chest/
Upper_arm.L/R drivers. A motion-space holdout can still fail even when
interpolation and contiguous validation improve; that failure is reported and
must not be hidden.

## Verification

```powershell
pwsh ./tools/verify.ps1
```

The canonical check runs Ruff, Pyright, PSScriptAnalyzer, Pester, and
Markdownlint when the npm tool is installed. Runtime Blender/Dem Bones
verification is intentionally separate because those third-party executables
are not vendored.

## Input contract and limitations

- Cloth topology must be fixed for every cache frame.
- FBX and Alembic vertex order must match exactly.
- The rest FBX must contain the same mesh represented by the first cache frame.
- The clean route currently expects a flat Dem Bones output; grouped joint
  hierarchies are rejected to avoid silently changing bind semantics.
- This PoC does not infer garment seams, semantic anatomy, or a universal
  skeleton mapping.
- A trained body-driven baseline is not the same as a VRChat runtime rig. For
  deployment, export helper bones into the avatar and connect them through
  allowed Unity/VRChat components. Custom runtime scripts are not assumed to
  be uploadable.
- Bone count is a quality/performance trade-off; 50 bones was the verified
  sample setting, not a universal recommendation.

## VRChat direction

VRChat currently allows VRC Constraints, Animator, VRC PhysBones, and
VRCPhysBoneColliders on avatars, while arbitrary custom components/scripts are
not an upload-safe dependency. The intended runtime split is body-driven
helper-bone motion for large pose-dependent folds plus PhysBones for secondary
motion. Prefer VRChat's own [Constraints](https://creators.vrchat.com/common-components/constraints/)
over Unity constraints for avatar content, and check the [allowed avatar
components](https://creators.vrchat.com/avatars/whitelisted-avatar-components/)
before shipping. See [`docs/architecture.md`](docs/architecture.md) for the
boundary between the offline exporter and that runtime design.

## License and attribution

The original code in this repository is MIT licensed. Dem Bones is an external
Electronic Arts BSD-3-Clause project. ClothTransformer data is not included;
obtain it directly from its official distribution, comply with CC BY 4.0 and
the listed upstream asset terms, and provide attribution in any experiment.

See [`docs/troubleshooting.md`](docs/troubleshooting.md) and
[`examples/README.md`](examples/README.md) for reproducible diagnostics and
legal synthetic fixtures.

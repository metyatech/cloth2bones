# PoC 3.1: body-local cloth drivers

PoC 3.1 removes the frame-wise rigid transform of the whole collider before
learning any cloth-bone motion. The input is therefore a body-local residual
motion signal, not a frame number and not a shared transform copied to every
cloth bone.

For each frame, the tool fits a Kabsch transform `G` from the rest collider to
the current collider and canonicalizes points with `inverse(G)`. Cloth-bone
targets are represented as a six-value local rigid transform per bone:

```text
rotation vector (x, y, z) + translation (x, y, z)
```

The automatic body regions use rest-space position plus PCA scores of the
canonical residual trajectories. Collision-edge connected-component counts
are reported so fragmented clusters are visible instead of silently being
treated as semantic anatomy. Region features contain rotation vector,
translation, centroid delta, and log singular values of a local deformation
fit.

## Run

The wrapper requires a clean FBX and a matching collider/cloth NPZ. It creates
all artifacts under the explicit output directory:

```powershell
pwsh ./tools/run_body_local_poc.ps1 `
  -Npz ./input/sim_00000.npz `
  -CleanFbx ./out/cloth_clean_rigid_50.fbx `
  -BlenderExe 'C:/Program Files/Blender Foundation/Blender/blender.exe' `
  -OutputRoot ./out/body_local_poc
```

The wrapper exports poses from the exact clean FBX supplied by `-CleanFbx`,
then runs the NumPy analysis, writes teacher/predicted FBX files, verifies the
predicted FBX against the canonical teacher, and renders five comparison
frames. `-SelectedModel` accepts `linear`, `ridge`, `polynomial`, `nearest`,
`rbf`, or `local-ridge`; `nearest` is the conservative verified baseline for
this sample. `-BoneCounts` controls the reduction benchmark, for example
`-BoneCounts 50,32,20,16,8`.

The lower-level analysis command is useful when Blender output already exists:

```powershell
python ./tools/body_local_poc.py `
  --npz ./input/sim_00000.npz `
  --poses ./out/clean_rig_poses.npz `
  --out-root ./out/body_local_poc `
  --selected-model nearest
```

## Evaluation protocol

The report evaluates static rest (the identity/global-only baseline), then
linear, ridge, degree-two polynomial, nearest body-feature pose, RBF, and
region-local ridge models. It includes:

- contiguous train frames 1-180 and validation frames 181-240;
- every-fifth and every-tenth interleaved holdouts;
- a body-feature motion-space cluster holdout;
- mean RMS, maximum per-frame RMS, maximum point error, and 95th percentile
  point error;
- per-bone motion variance and body-feature correlation inspection;
- 50/32/20/16/8-bone reduction measurements.

No frame index, time, normalized progress, cloth vertex, or cloth teacher
feature enters the model. The motion-space result is deliberately retained as
a failure check: nearest-pose interpolation can improve interpolation and
contiguous extrapolation while still failing a genuinely unseen body-feature
cluster. This PoC must not be described as a universal humanoid or VRChat
runtime cloth solver.

## Output contract

Important files are:

```text
body_local_report.json
body_local_features.npz
teacher_local_poses_50.npz
predicted_local_poses_50.npz
bone_mapping.json
body_local_teacher_50.fbx
body_local_predicted_50.fbx
predicted_verify.json
body_local_comparison.blend
previews/body_local_comparison_*.png
```

The pose NPZ and target FBX must come from the same clean rig. The Blender
application tool compares rest vertices and normalized vertex weights before
writing animation and rejects mismatches. This prevents a subtle failure mode
where identically named bones have different weight assignments.

## Interpretation and VRChat bridge

The automatic regions are collider-derived and have no semantic left/right
joint labels. A production training set should replace them with stable
humanoid drivers such as Hips, Spine, Chest, Upper_arm.L/R, and
Lower_arm.L/R, or learn a fixed correspondence from an explicitly annotated
body. The offline mapping can then be baked into helper-bone animation or
approximated with supported Unity/VRChat constraints and Animator state. The
large pose-dependent fold belongs in body-driven helper bones; PhysBones are a
separate option for secondary jiggle and lag.

The current sample is a single 240-frame motion. Passing contiguous and
interleaved validation does not establish arbitrary-pose generalization. New
body poses, held-out motion families, and semantic skeleton features are still
required before applying the method to production garments.

# Architecture

## Offline conversion

The converter has four explicit boundaries:

1. Dem Bones reads a rest FBX and animated Alembic and solves a low-rank LBS
   decomposition.
2. `build_clean_blender_rig.py` reads the Dem Bones weights and flat bones,
   creates a new identity-space Blender armature, and fits one rigid transform
   per bone against each cache frame with weighted Kabsch iterations.
3. Blender exports a new FBX, keeping bind pose, vertex groups, and animation
   in the same scene and coordinate convention.
4. The verification scripts re-import the FBX and compare same-index evaluated
   vertices against the original cache.

The raw Dem Bones FBX is an intermediate diagnostic. It is not the canonical
runtime artifact for Blender/Unity input where the standard writer's local
animation/bind transform conventions do not agree with the source scene.

## Body Motion PoC

The NPZ collider animation is reduced to four deterministic driver regions:
the full collider, torso, positive-X side, and negative-X side. Each region is
fit from rest to each frame with a rigid Kabsch transform. The transform's
rotation delta and translation become the feature vector.

Clean FBX poses are exported to an NPZ. A ridge-linear mapping is trained on an
initial contiguous set of frames and evaluated on later held-out frames. The
predicted rotation blocks are projected back to SO(3), then applied with the
existing weights. Because the model never receives the frame index, a held-out
score measures whether body motion contains useful predictive information.

This is deliberately a research baseline. It is not yet a semantic mapping
from `UpperArm.L` to named cloth bones because the input NPZ has collider
vertices but no skeleton/joint transform stream.

## Runtime target

The eventual avatar artifact should contain a small fixed helper hierarchy,
with body-driven transforms represented by Unity/VRChat-supported constraints
or animator parameters. A separate PhysBone chain may add secondary motion.
The offline learned model is useful for deciding which helper bones and
weights to author; it is not itself a custom runtime component.

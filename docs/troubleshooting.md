# Troubleshooting

## Mesh explodes in Pose Position

Check the clean FBX rather than the raw Dem Bones FBX. If Rest Position is
correct but Pose Position is broken, the failure is in the raw writer's
bind/local animation space, not in the cloth topology. Re-run the clean route
and inspect the Blender acceptance report.

## Vertex count mismatch

The rest FBX and Alembic must have the same vertex count and same index order.
Use `compare_fbx_to_abc.py` at frame 1 before increasing the bone count.

## Body PoC rejects rest vertices

`dump_blender_rig_poses.py` must read the same clean FBX generated from
`traj[0]`. A different unit scale or a mesh with unapplied object transforms
will fail the explicit rest-vertex check. Fix the source convention instead of
adding a hidden scale in the regression script.

## Pose NPZ weights do not match the target FBX

Export `dump_blender_rig_poses.py` from the exact clean FBX passed to
`apply_body_driven_poses.py`. Do not reuse a pose NPZ from another clean-rig
export merely because its armature has the same bone names. The application
tool compares normalized vertex weights as well as rest vertices and rejects a
mismatch before producing an FBX.

## Body validation is worse than the rest baseline

Inspect the driver-region masks and the train/validation split. The current
driver is geometric, not semantic; it can be insufficient when a collider
contains multiple disconnected bodies or when the held-out motion leaves the
training range. Do not report a frame-index lookup as a successful body model.

## Unity import

Use a Generic rig for the generated helper-bone FBX. Confirm the importer scale
factor matches the scale convention used during Blender export and verify the
clip length and bone count before judging animation quality.

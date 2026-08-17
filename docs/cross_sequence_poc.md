# PoC 3.2: cross-sequence body-driven cloth prediction

PoC 3.2 tests a stricter question than PoC 3.1:

```text
complete Body Motion sequence A+B+C -> training
complete unseen Body Motion sequence D -> test
```

The test sequence contributes no cloth teacher vertices, bone targets, nearest
database entries, or sequence identifier to model fitting. The model input is
fixed rest-space body-region motion after frame-wise whole-body Kabsch
canonicalization. The target is a six-value local transform for each cloth
bone: rotation vector plus translation.

## Dataset audit result

The official [ClothTransformer dataset card](https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset)
currently describes 56 Human Garment trajectories, 240 frames per trajectory,
and T-shirts and skirts. An audit of the 56 released NPZ files found 54 cloth
topology groups and 54 exact rest-geometry groups. The only repeated groups
were `sim_00004`/`sim_00005` and `sim_00036`/`sim_00037`; their complete NPZ
files were byte-identical. Therefore this release does not provide a
non-duplicate same-garment, different-motion Group A/B for this experiment.

The audit records every field, shape, dtype, cloth/body topology hash, edge
hash, rest hash, and frame count. It does not copy the dataset into this
repository. The dataset is CC BY 4.0 with third-party asset terms; cite and
comply with the official card when using it.

## Reproducible fallback

[CLOTH3D](https://hbertiche.github.io/CLOTH3D/) and
[CAPE](https://cape.is.tue.mpg.de/dataset.html) have suitable research
descriptions, but their official data access requires registration and/or
acceptance of data-specific terms. They are not silently downloaded or
repackaged by this OSS tool. The checked-in fallback generator creates a
controlled benchmark from an existing local rest cloth and clean-rig weight
file. It generates four sequences with exactly one shared topology, weights,
and bone indexing:

- `seq_A_left_raise`
- `seq_B_right_raise`
- `seq_C_alternating_swing`
- `seq_D_combined_holdout`

The first three are training folds and D is the primary full-sequence holdout.
The synthetic teacher is an analytic, fixed body-region mapping with a small
velocity term. It validates cross-sequence feature/model/rig contracts; it is
not evidence that a physical cloth solver has generalized to arbitrary human
motion.

Run the complete fallback benchmark with:

```powershell
pwsh ./tools/run_cross_sequence_poc.ps1 `
  -ReferenceNpz ./input/sim_00000.npz `
  -CommonPoseNpz ./out/body_local_poc/clean_rig_poses.npz `
  -SyntheticRoot ./out/cross_sequence_data/synthetic_same_rig `
  -OutputRoot ./out/cross_sequence_poc
```

To audit a downloaded official subset without running the model:

```powershell
python ./tools/audit_cross_sequence_dataset.py `
  --root ./input/clothtransformer_human_garment `
  --out ./out/cross_sequence_poc/dataset_inventory.json
```

The lower-level steps are `generate_cross_sequence_synthetic.py`,
`audit_cross_sequence_dataset.py`, `cross_sequence_poc.py`, and
`plot_cross_sequence_results.py`.

## Model and leakage protocol

The tool evaluates linear, ridge, degree-two polynomial, nearest pose, RBF,
region-local ridge, and velocity-augmented ridge/nearest models. It performs
four leave-one-sequence-out folds and reports static-rest/global-only baseline,
mean RMS, maximum frame RMS, maximum point error, 95th percentile point error,
per-frame errors, and nearest-train-feature distance. It also compares
50/32/20/16/8 helper-bone reductions.

The following are explicit negative checks:

- frame number, time, progress, and sequence ID are absent from features;
- cloth teacher data is absent from features and nearest lookup;
- test frames are absent from model fitting;
- common body regions are fitted from rest geometry only;
- global body rigid motion is removed before feature extraction.

Velocity features are a separate comparison, not a hidden input. On the current
synthetic fixture they are worse than pose-only ridge because the fixture's
velocity term is intentionally small and the added feature dimensionality is
not regularized enough; the report keeps that result visible.

## Blender artifacts

After NumPy evaluation, use the generated pose NPZ files with the existing
clean-rig Blender application tool. The resulting Teacher, Research, Runtime,
and Rest FBX files can be imported together. The comparison renderer creates
Teacher / Predicted / Rest / Error panels at five holdout frames and saves a
reviewable `.blend` file. All generated data stays outside the repository.

## Interpretation and production gap

The current successful cross-sequence result is a controlled synthetic
generalization test. It does not prove that the Human Garment sample or a
Marvelous Designer garment can be driven by arbitrary VRChat humanoid poses.
The next production-grade dataset must contain the same garment topology and
correspondence over multiple distinct body-motion sequences, preferably with
semantic body joint transforms. The region IDs should then be replaced by
Hips/Spine/Chest/Upper_arm.L/R/Lower_arm.L/R drivers and the learned mapping
reduced to supported Animator, VRC Constraints, and PhysBones components.

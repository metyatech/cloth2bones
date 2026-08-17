# Examples

This directory contains only a small synthetic NPZ generator. It creates no
third-party mesh or dataset and is safe to run in CI or a clean checkout:

```powershell
python ./examples/generate_synthetic_npz.py --out ./out/synthetic.npz
python ./tools/inspect_npz.py ./out/synthetic.npz
```

The real cloth conversion requires external Dem Bones, Blender, and a fixed
topology rest FBX/Alembic pair. Keep those files outside the repository or in a
local ignored `input/` directory.

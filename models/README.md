# Sentinel model files

This directory is the runtime location for Sentinel's model weights and generated
inference artifacts. The artifacts are not included in this Git repository, so a
fresh clone will contain this README but not the model files.

See [`../MODEL_ASSETS.md`](../MODEL_ASSETS.md) for the required filenames,
availability notes, and bundle verification instructions. Python packages belong
in `requirements.txt`; model weights must be obtained separately and placed here.

After installing the model bundle, verify it from the repository root with:

```bash
python scripts/verify_model_bundle.py --with-rl
```

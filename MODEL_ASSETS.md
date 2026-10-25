# Sentinel model assets

The files below are runtime artifacts, not Python packages. `pip install -r requirements.txt`
installs libraries only; it cannot install model weights or memory banks. The trained bundle
is intentionally excluded from this Git repository because it is approximately 834 MB locally.
`MODEL_ASSETS.sha256` records the expected SHA-256 digest for each required model and calibration file.

## Pipeline B inference bundle

Place these files under `models/` beside `app.py`, or set `MVTEC_PROJECT_DIR` to a directory
containing both `models/` and `assets/`. The authoritative paths are also listed in
[`model_manifest.json`](model_manifest.json).

- `yolo26n.pt`
- `wide_resnet50_2_patch_extractor.pth`
- `stfpm_teacher_resnet18_grid.pth`
- `stfpm_student_resnet18_grid.pth`
- `yolo26_embedding_knn_<category>.joblib` for each supported category
- `patchcore_patch_bank_<category>.pt` for each supported category
- `causal_temporal_transformer.pt` and `temporal_transformer_input_scaler.joblib` for temporal inference

Optional subtype files are documented in `model_manifest.json`. The six small active calibration
arrays are already included under `assets/`; model weights and generated caches remain ignored.

## Artifact distribution checklist

No authorized public download URL is configured in this project. Before advertising a ready-to-run
model bundle, the project owners should publish it to an approved release or artifact host, record
the exact version and download URL here. After downloading, run
`python scripts/verify_model_bundle.py --with-rl` to check presence and SHA-256 against the tracked
manifest. You can also check files with `shasum -a 256 <file>` (macOS/Linux) or
`Get-FileHash <file> -Algorithm SHA256` (PowerShell). Do not place credentials or signed private
download URLs in this repository.

Pipeline A's integrated inference export and Q-table are not present. A PPO checkpoint is also not
present; the app labels its adaptive video fallback. These components are not made available by
installing Python requirements.

## Data and redistribution

MVTec AD itself is not included. Obtain it from its official source and follow its dataset terms.
Before publishing the bundled illustrative images and derived video clips, confirm that their
redistribution is permitted. The repository currently includes them as application examples;
replace or remove them before a public release if rights are unclear.

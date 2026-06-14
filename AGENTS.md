# TRELLIS Modal App

## Commands
- `modal run src/trellis_app/app.py --prompt "a chair"` — generate a 3D model and upload to R2
- `uv run trellis-generate --prompt "a chair"` — local wrapper (calls modal run)
- `modal deploy src/trellis_app/app.py` — deploy as persistent app

## Secrets
- `r2-credentials` — Modal secret with R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT, R2_PUBLIC_URL

## Model cache
- A Modal Volume named `trellis-models` caches downloaded HuggingFace weights.
- Without it, every cold start re-downloads ~14GB.
- Create it with: `modal volume create trellis-models`

## GPU
- Default: A100 (80GB). Sufficient for text-xlarge (2B params).
- For smaller models use: `--gpu "A10G"` or `--gpu "L40S"`.

## R2 Egress
- GLB files are uploaded to the configured R2 bucket.
- The public URL is returned so you can download/share.
- Bucket must have public access enabled in R2 dashboard.

## TRELLIS build
The Modal image clones https://github.com/microsoft/TRELLIS at build time
and installs all CUDA extensions (xformers, flash-attn, diffoctreerast, etc.).
First build takes ~15–20 minutes. Subsequent builds are cached by Modal.

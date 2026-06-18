# TRELLIS.2 Modal App

## Commands
- `modal run src/trellis_app/app.py --prompt "a dragon statue"` — generate a 3D model and upload to R2
- `uv run trellis-generate --prompt "a dragon statue"` — local wrapper (calls modal run)
- `modal deploy src/trellis_app/app.py` — deploy as persistent app

## How it works
1. **FLUX.1-schnell** (text-to-image, 4 steps) generates an image from the prompt.
2. **TRELLIS.2-4B** (4B param image-to-3D) generates geometry + PBR materials.
3. GLB is exported via **o-voxel** and uploaded to Cloudflare R2.

## Secrets
- `r2-credentials` — Modal secret with R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT, R2_PUBLIC_URL

## Model cache
- A Modal Volume named `trellis-models` caches downloaded HuggingFace weights.
- FLUX (~8GB) + TRELLIS.2 (~16GB) are cached, avoiding re-downloads.
- Create it with: `modal volume create trellis-models`

## GPU
- **Default: L40S (48GB)** — ~50% cheaper than A100, enough VRAM for FLUX + TRELLIS.
- Override with `--gpu A100` or `--gpu H100` (for 1536 resolution).
- TRELLIS.2-4B needs ~24GB GPU memory; FLUX.1-schnell adds ~12GB in bf16.
- FLUX is loaded once at container start and CPU-offloaded during 3D generation.

## R2 Egress
- GLB files are uploaded to the configured R2 bucket with PBR materials.
- The public URL is returned so you can download/share.
- Bucket must have public access enabled in R2 dashboard.

## Image build
The Modal image builds TRELLIS.2 from source:
1. Base: `nvidia/cuda:12.4.0-devel-ubuntu22.04` (provides CUDA 12.4 toolkit)
2. PyTorch 2.6.0 + CUDA 12.4
3. flash-attn 2.7.3, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel
4. First build takes ~20-30 minutes. Subsequent builds are cached by Modal.

## Resolutions
- `512` — ~3s on H100, fastest, lower quality
- `1024` (default) — ~17s on H100, good balance
- `1536` — ~60s on H100, best quality, requires more VRAM

## CLI options
- `--prompt` (required) — text prompt
- `--gpu` — GPU type (default: L40S)
- `--resolution` — TRELLIS resolution: 512, 1024, 1536 (default: 512)
- `--texture-size` — PBR texture resolution: 512 or 1024 (default: 1024; 512 is 4x faster)
- `--decimation-target` — target triangle count (default: 10000)
- `--deploy` — deploy as persistent app before running

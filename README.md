# TRELLIS.2 Modal App

Generate 3D models from text prompts using [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (4B param image-to-3D) on [Modal](https://modal.com), with automatic Cloudflare R2 upload for egress.

## Pipeline

```
Text Prompt → FLUX.1-schnell (4 steps) → Image → TRELLIS.2-4B → PBR Mesh → GLB → Cloudflare R2
```

## Setup

```bash
# 1. Install Modal CLI
pip install modal

# 2. Create the model cache volume (avoids re-downloading ~24GB of weights)
modal volume create trellis-models

# 3. Create secrets for R2 (S3-compatible storage)
modal secret create r2-credentials \
  R2_ACCESS_KEY_ID=<key> \
  R2_SECRET_ACCESS_KEY=<secret> \
  R2_BUCKET_NAME=trellis-output \
  R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
  R2_PUBLIC_URL=https://pub-<hash>.r2.dev
```

## Usage

```bash
# Generate a 3D model
modal run src/trellis_app/app.py --prompt "a dragon statue"

# Or using the local wrapper
uv run trellis-generate --prompt "a dragon statue"

# Specify resolution (512 / 1024 / 1536)
modal run src/trellis_app/app.py --prompt "a chair"  # uses default 1024

# Deploy as a persistent app
modal deploy src/trellis_app/app.py
```

The command outputs a download URL for the GLB file with PBR materials.

## GPU Requirements

| GPU | Memory | Works? |
|-----|--------|--------|
| A100 80GB | 80GB | Yes (default) |
| H100 | 80GB | Yes |
| A100 40GB | 40GB | Possibly at 512 resolution |
| A10G / L40S | 24GB | Minimum for 512 res, may OOM |

TRELLIS.2-4B requires ~24GB GPU memory. An **A100 80GB or H100** is recommended.

## Performance (on H100)

| Resolution | Time | Quality |
|------------|------|---------|
| 512 | ~3s | Fast, lower quality |
| 1024 | ~17s | Default, good balance |
| 1536 | ~60s | Best quality, more VRAM |

## Cloudflare R2

The GLB is uploaded to your R2 bucket with a public URL. Enable public access in the R2 dashboard for the bucket so the URL is downloadable.

## Image Build

The first Modal run builds a container image with:
- **Base:** `nvidia/cuda:12.4.0-devel-ubuntu22.04` (CUDA 12.4 toolkit)
- **PyTorch:** 2.6.0 + CUDA 12.4
- **CUDA extensions:** flash-attn 2.7.3, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel

First build takes ~20–30 minutes. Subsequent runs are instant (cached by Modal).

## License

MIT

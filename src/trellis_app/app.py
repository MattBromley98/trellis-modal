import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from trellis_app.r2 import upload_glb

TRELLIS2_REPO = "https://github.com/microsoft/TRELLIS.2.git"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "ninja-build", "cmake", "libjpeg-dev",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libgles2",
        "libglvnd0", "libxkbcommon0", "libsm6", "libxext6",
    )
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 extra_options=["--index-url", "https://download.pytorch.org/whl/cu124"])
    .pip_install(
        "imageio", "imageio-ffmpeg", "tqdm", "easydict",
        "opencv-python-headless", "ninja", "trimesh", "transformers",
        "kornia", "timm", "zstandard",
    )
    .run_commands(
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    .run_commands(
        f"git clone {TRELLIS2_REPO} /trellis2_src --recursive",
    )
    .run_commands(
        "pip install /trellis2_src/o-voxel --no-build-isolation",
        "pip install flash-attn==2.7.3",
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && pip install /tmp/nvdiffrast --no-build-isolation",
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec && pip install /tmp/nvdiffrec --no-build-isolation",
        "git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive && pip install /tmp/CuMesh --no-build-isolation",
        "git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive && pip install /tmp/FlexGEMM --no-build-isolation",
    )
    .env({"PYTHONPATH": "/trellis2_src"})
    .pip_install("modal", "boto3", "Pillow", "diffusers")
)

app = modal.App("trellis-3d", image=image)

model_volume = modal.Volume.from_name("trellis-models", create_if_missing=True)

HUGGINGFACE_CACHE = "/hf-cache"


@app.cls(
    gpu="A100",
    timeout=900,
    secrets=[modal.Secret.from_name("r2-credentials")],
    volumes={HUGGINGFACE_CACHE: model_volume},
    scaledown_window=60,
)
class TrellisGenerator:
    def __init__(self):
        os.environ["HF_HOME"] = HUGGINGFACE_CACHE
        os.environ["HF_HUB_CACHE"] = f"{HUGGINGFACE_CACHE}/hub"
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.pipeline = None

    @modal.enter()
    def load_pipeline(self):
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
        )
        self.pipeline.cuda()

    @modal.method()
    def generate(self, prompt: str, seed: int = 42) -> str:
        import gc
        import tempfile

        import torch
        from diffusers import FluxPipeline

        import o_voxel

        out_dir = Path(tempfile.mkdtemp())

        print(f"[1/3] Generating image from prompt: {prompt!r}")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        gen = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=gen,
        ).images[0]
        image.save(str(out_dir / "input.png"))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        print("[2/3] Generating 3D geometry + PBR materials...")
        mesh = self.pipeline.run(image, seed=seed, preprocess_image=True)[0]
        mesh.simplify(2_000_000)

        print("[3/3] Exporting GLB...")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1_000_000,
            texture_size=4096,
            remesh=False,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        safe_name = prompt.replace(" ", "_")[:64]
        local_path = out_dir / f"{safe_name}.glb"
        glb.export(str(local_path), extension_webp=False)

        print(f"Uploading to R2...")
        url = upload_glb(local_path, prompt)
        print(f"Uploaded to: {url}")

        return url


@app.local_entrypoint()
def main(prompt: str = "a chair"):
    generator = TrellisGenerator()
    url = generator.generate.remote(prompt)
    print(f"\nDownload URL: {url}")

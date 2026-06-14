import os
import sys
from pathlib import Path

import modal

from .generate import generate_glb
from .r2 import upload_glb

TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", "cmake",
        "cuda-toolkit-12-4",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
        "libxrender-dev", "libgomp1", "libegl1", "libxkbcommon0",
        "libegl1-mesa", "libgles2", "libglvnd0",
    )
    .pip_install("torch==2.4.0", "torchvision==0.19.0",
                 "--index-url", "https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.27.post2",
                 "--index-url", "https://download.pytorch.org/whl/cu121")
    .pip_install("flash-attn==2.6.3")
    .run_commands(
        f"git clone --recurse-submodules {TRELLIS_REPO} /trellis",
    )
    .pip_install(
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
        "easydict",
        "opencv-python-headless",
        "scipy",
        "ninja",
        "trimesh",
        "open3d",
        "xatlas",
        "pyvista",
        "pymeshfix",
        "igraph",
        "transformers",
        "rembg",
        "onnxruntime",
    )
    .run_commands(
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "pip install spconv-cu120",
        "pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
        "git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && pip install /tmp/nvdiffrast",
        "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && pip install /tmp/diffoctreerast",
        "git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/",
    )
    .env({"PYTHONPATH": "/trellis"})
    .pip_install("modal", "boto3", "Pillow")
)

app = modal.App("trellis-3d", image=image)

model_volume = modal.Volume.from_name("trellis-models", create_if_missing=True)

HUGGINGFACE_CACHE = "/hf-cache"


@app.cls(
    gpu="A100",
    timeout=600,
    secrets=[modal.Secret.from_name("r2-credentials", required=True)],
    volumes={HUGGINGFACE_CACHE: model_volume},
    container_idle_timeout=60,
)
class TrellisGenerator:
    def __init__(self):
        os.environ["SPCONV_ALGO"] = "native"
        os.environ["HF_HOME"] = HUGGINGFACE_CACHE
        os.environ["HF_HUB_CACHE"] = f"{HUGGINGFACE_CACHE}/hub"
        self.model_name = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-text-xlarge")

    @modal.enter()
    def load_pipeline(self):
        from trellis.pipelines import TrellisTextTo3DPipeline
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_name)
        self.pipeline.cuda()

    @modal.method()
    def generate(self, prompt: str, seed: int = 42) -> str:
        import tempfile
        from trellis.utils import postprocessing_utils

        print(f"Generating 3D model for prompt: {prompt!r}")

        outputs = self.pipeline.run(prompt, seed=seed)

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )

        out_dir = Path(tempfile.mkdtemp())
        safe_name = prompt.replace(" ", "_")[:64]
        local_path = out_dir / f"{safe_name}.glb"
        glb.export(str(local_path))

        print(f"Uploading {local_path} to R2...")
        url = upload_glb(local_path, prompt)
        print(f"Uploaded to: {url}")

        return url


@app.local_entrypoint()
def main(prompt: str = "a chair"):
    generator = TrellisGenerator()
    url = generator.generate.remote(prompt)
    print(f"\nDownload URL: {url}")

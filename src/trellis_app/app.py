import hashlib
import os
from pathlib import Path

import modal


def _upload_glb(local_path: str | Path, prompt: str) -> str:
    import boto3
    path = Path(local_path)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    key = f"trellis/{prompt_hash}/{path.name}"
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.upload_file(str(path), os.environ["R2_BUCKET_NAME"], key,
                   ExtraArgs={"ContentType": "model/gltf-binary"})
    return f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{key}"

TRELLIS2_REPO = "https://github.com/microsoft/TRELLIS.2.git"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "ninja-build", "cmake", "g++", "libjpeg-dev",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libgles2",
        "libglvnd0", "libxkbcommon0", "libsm6", "libxext6",
    )
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy", "imageio", "imageio-ffmpeg", "tqdm", "easydict",
        "opencv-python-headless", "ninja", "trimesh", "transformers==4.48.2",
        "kornia", "timm==1.0.14", "zstandard", "plyfile", "wheel", "setuptools",
    )
    .run_commands(
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    .run_commands(
        f"git clone {TRELLIS2_REPO} /trellis2_src --recursive",
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast",
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec",
        "git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive",
        "git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive",
    )
    .run_commands(
        "sed -i 's|pipeline.rembg_model = getattr(rembg,.*|pipeline.rembg_model = None|' /trellis2_src/trellis2/pipelines/trellis2_image_to_3d.py",
        "sed -i 's|self.model.layer|self.model.encoder.layer|' /trellis2_src/trellis2/modules/image_feature_extractor.py",
    )
    .run_commands(
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;9.0' CUDA_HOME=/usr/local/cuda-12.4 pip install /tmp/CuMesh --no-build-isolation",
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;9.0' CUDA_HOME=/usr/local/cuda-12.4 pip install /tmp/FlexGEMM --no-build-isolation",
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;9.0' CUDA_HOME=/usr/local/cuda-12.4 pip install /tmp/nvdiffrast --no-build-isolation",
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;9.0' CUDA_HOME=/usr/local/cuda-12.4 pip install /tmp/nvdiffrec --no-build-isolation",
    )
    .run_commands(
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;9.0' CUDA_HOME=/usr/local/cuda-12.4 pip install /trellis2_src/o-voxel --no-build-isolation",
        "pip install flash-attn==2.7.3",
    )
    .env({"PYTHONPATH": "/trellis2_src"})
    .pip_install("modal", "boto3", "Pillow", "diffusers", "accelerate")
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
    @modal.enter()
    def load_pipeline(self):
        os.environ["HF_HOME"] = HUGGINGFACE_CACHE
        os.environ["HF_HUB_CACHE"] = f"{HUGGINGFACE_CACHE}/hub"
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        mesh = self.pipeline.run(image, seed=seed, preprocess_image=False)[0]
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
        url = _upload_glb(local_path, prompt)
        print(f"Uploaded to: {url}")

        return url


@app.local_entrypoint()
def main(prompt: str = "a chair"):
    generator = TrellisGenerator()
    url = generator.generate.remote(prompt)
    print(f"\nDownload URL: {url}")

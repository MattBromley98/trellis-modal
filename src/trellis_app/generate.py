import os
import gc
import tempfile
from pathlib import Path

import torch
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


TTI_MODEL = "black-forest-labs/FLUX.1-schnell"
TRELLIS2_MODEL = "microsoft/TRELLIS.2-4B"


def text_to_image(prompt: str, seed: int = 42) -> Image.Image:
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(TTI_MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator,
    ).images[0]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return image


def generate_glb(
    prompt: str,
    seed: int = 42,
    resolution: str = "1024",
    simplify_faces: int = 2_000_000,
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir or tempfile.mkdtemp())

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS2_MODEL)
    pipeline.cuda()

    image = text_to_image(prompt, seed=seed)
    image.save(str(output_dir / "input.png"))

    mesh = pipeline.run(
        image,
        seed=seed,
        preprocess_image=True,
        pipeline_type=resolution,
    )[0]
    mesh.simplify(simplify_faces)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    safe_name = prompt.replace(" ", "_")[:64]
    out_path = output_dir / f"{safe_name}.glb"
    glb.export(str(out_path), extension_webp=False)

    del pipeline, mesh, glb
    gc.collect()
    torch.cuda.empty_cache()

    return out_path

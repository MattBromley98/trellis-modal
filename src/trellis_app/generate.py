import os
import tempfile
from pathlib import Path

os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils


def generate_glb(
    prompt: str,
    model_name: str = "microsoft/TRELLIS-text-xlarge",
    seed: int = 42,
    simplify: float = 0.95,
    texture_size: int = 1024,
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir or tempfile.mkdtemp())

    pipeline = TrellisTextTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()

    outputs = pipeline.run(
        prompt,
        seed=seed,
    )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify,
        texture_size=texture_size,
    )

    safe_name = prompt.replace(" ", "_")[:64]
    out_path = output_dir / f"{safe_name}.glb"
    glb.export(str(out_path))
    return out_path

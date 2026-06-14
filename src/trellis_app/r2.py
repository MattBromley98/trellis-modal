import os
import hashlib
from pathlib import Path

import boto3


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_glb(local_path: str | Path, prompt: str) -> str:
    path = Path(local_path)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    key = f"trellis/{prompt_hash}/{path.name}"

    s3 = _get_s3_client()
    bucket = os.environ["R2_BUCKET_NAME"]
    s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "model/gltf-binary"})

    public_url = os.environ["R2_PUBLIC_URL"].rstrip("/")
    return f"{public_url}/{key}"

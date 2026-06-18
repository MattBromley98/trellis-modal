import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D models from text prompts using TRELLIS on Modal"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt for 3D generation",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type (e.g., L40S, A100, H100). Default: L40S",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="TRELLIS resolution: 512, 1024, or 1536. Default: 512",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=None,
        help="PBR texture resolution (512 or 1024). Default: 1024",
    )
    parser.add_argument(
        "--decimation-target",
        type=int,
        default=None,
        help="Target triangle count. Default: 10000",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy as a persistent Modal app first",
    )
    args = parser.parse_args()

    app_path = str(Path(__file__).resolve().parent / "app.py")

    if args.gpu:
        os.environ["MODAL_GPU"] = args.gpu

    modal_args = ["modal", "run"]

    if args.deploy:
        print("Deploying app...")
        subprocess.run(["modal", "deploy", app_path], check=True)

    modal_args.append(app_path)
    modal_args.extend(["--prompt", args.prompt])

    if args.resolution:
        modal_args.extend(["--resolution", args.resolution])
    if args.texture_size:
        modal_args.extend(["--texture-size", str(args.texture_size)])
    if args.decimation_target:
        modal_args.extend(["--decimation-target", str(args.decimation_target)])

    sys.exit(subprocess.run(modal_args).returncode)


if __name__ == "__main__":
    main()

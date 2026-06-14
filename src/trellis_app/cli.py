import argparse
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
        help="GPU type (e.g., A100, A10G, L40S). Default: A100",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy as a persistent Modal app first",
    )
    args = parser.parse_args()

    app_path = str(Path(__file__).resolve().parent / "app.py")

    modal_args = ["modal", "run"]

    if args.deploy:
        print("Deploying app...")
        subprocess.run(["modal", "deploy", app_path], check=True)

    if args.gpu:
        modal_args.extend(["-e", f"MODAL_GPU={args.gpu}"])

    modal_args.append(app_path)
    modal_args.extend(["--prompt", args.prompt])

    sys.exit(subprocess.run(modal_args).returncode)


if __name__ == "__main__":
    main()

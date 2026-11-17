"""Set up and run the publication experiments with one command."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def call(*args):
    subprocess.run([str(arg) for arg in args], cwd=REPO, check=True)


def restart_with_conda(parser):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        parser.error("Use Python 3.12, 3.13 or 3.14, or install Conda for automatic setup.")
    env = HERE / ".conda-python312"
    python = env / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        current = ".".join(map(str, sys.version_info[:3]))
        print(f"Python {current} is active. Setting up Python 3.12 with Conda in {env}.", flush=True)
        call(conda, "create", "--yes", "--prefix", env, "--override-channels",
             "--channel", "conda-forge", "python=3.12", "pip")
    call(conda, "run", "--prefix", env, "--no-capture-output", "python",
         Path(__file__).resolve(), *sys.argv[1:])


def gpu_present():
    command = shutil.which("nvidia-smi")
    if not command and sys.platform == "win32":
        command = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if not command.is_file():
            return False
    if not command:
        return False
    return subprocess.run([str(command), "-L"], capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run the short verification experiment.")
    parser.add_argument("--extended", action="store_true", help="Run the additional FLAURA and Byz-Clip21 comparison using existing paired plans.")
    parser.add_argument("--paper", action="store_true", help="Replay the paper's 240 metric jobs on the original Apple GPU backend.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare dependencies and reference PDFs without training.")
    args = parser.parse_args()
    if sum((args.extended, args.paper, args.setup_only)) > 1 or args.paper and args.smoke:
        parser.error("Choose one of --paper, --extended or --setup-only; --paper uses the full 48-round protocol.")
    if sys.version_info[:2] not in ((3, 12), (3, 13), (3, 14)):
        restart_with_conda(parser)
        return

    device = args.device
    if device == "auto" and gpu_present():
        device = "cuda"
    cuda = device == "cuda"
    env = HERE / (".venv-cuda" if cuda else ".venv")
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    ready = env / "fedretrust-dependencies-v1"
    if not ready.exists():
        print(f"Installing the experiment dependencies in {env}", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
        torch_args = ["--index-url", "https://download.pytorch.org/whl/cu130"] if cuda else []
        if not cuda and sys.platform != "darwin":
            torch_args = ["--index-url", "https://download.pytorch.org/whl/cpu"]
        call(python, "-m", "pip", "install", *torch_args, "torch==2.13.0", "torchvision==0.28.0")
        call(python, "-m", "pip", "install", "numpy==2.5.2", "pandas==3.0.5")
        ready.touch()

    # The original RFA and Viceroy constructors record the source-paper hashes.
    call(python, REPO / "tools" / "prepare_reference_papers.py", "--download")
    if args.setup_only:
        print(f"Setup complete. Python: {python}", flush=True)
        return
    if args.paper:
        call(python, HERE / "replay_paper.py", "--device", device)
        return
    script = "extended_study.py" if args.extended else "study.py"
    call(python, HERE / script, "--device", device, *(["--smoke"] if args.smoke else []))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("A setup or training step stopped. See the error above. "
              "Run the same command after fixing the issue to resume.", file=sys.stderr)
        sys.exit(error.returncode or 1)

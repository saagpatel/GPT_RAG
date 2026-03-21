from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAURI_ROOT = REPO_ROOT / "apps" / "desktop" / "src-tauri"
BINARIES_DIR = TAURI_ROOT / "binaries"
BUILD_ROOT = TAURI_ROOT / ".sidecar-build"
ENTRYPOINTS = {
    "gpt-rag-api": REPO_ROOT / "scripts" / "desktop_sidecars" / "api_entry.py",
    "gpt-rag-worker": REPO_ROOT / "scripts" / "desktop_sidecars" / "worker_entry.py",
}


def host_target() -> str:
    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def build_sidecar(name: str, entrypoint: Path, target_triple: str) -> None:
    output_name = f"{name}-{target_triple}"
    work_dir = BUILD_ROOT / name
    spec_dir = work_dir / "spec"
    temp_dir = work_dir / "work"
    dist_dir = BINARIES_DIR

    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name",
            output_name,
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(temp_dir),
            "--specpath",
            str(spec_dir),
            "--paths",
            str(REPO_ROOT / "src"),
            str(entrypoint),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def main() -> None:
    target_triple = host_target()
    if BINARIES_DIR.exists():
        shutil.rmtree(BINARIES_DIR)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)

    for name, entrypoint in ENTRYPOINTS.items():
        build_sidecar(name, entrypoint, target_triple)

    print(f"Built desktop sidecars for {target_triple} in {BINARIES_DIR}")


if __name__ == "__main__":
    main()

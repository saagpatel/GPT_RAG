from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = REPO_ROOT / "apps" / "desktop"


def main() -> None:
    subprocess.run([sys.executable, "scripts/build_desktop_sidecars.py"], check=True, cwd=REPO_ROOT)
    subprocess.run(["npm", "run", "tauri:build"], check=True, cwd=DESKTOP_ROOT)


if __name__ == "__main__":
    main()

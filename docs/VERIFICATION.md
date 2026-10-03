# Developer verification

Run commands from the repository root in a disposable development checkout. Python 3.11+ is required. The fixture tests use temporary `GPT_RAG_HOME` directories and fake model runtimes; they do not require Ollama, downloaded models, or your document library.

## Python fixture lane

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests/test_answer_generation.py -q
python -m ruff check src tests
```

Use a relevant `tests/test_*.py` file or `-k` filter for the changed behavior. The broader source test suite is `python -m pytest`; CI attempts this development install but permits fallback to a base install. `python -m ruff format --check src tests` checks formatting without rewriting files. For a package-build check, use `python -m pip wheel --no-deps --wheel-dir dist .`; this builds a wheel and may download build dependencies.

There is no committed Python `uv.lock` in the current source. If a machine-local `.codex/verify.commands` harness is present, inspect its additional gates before running it. Its `--locked` setup requires an existing local lock: in a fresh disposable checkout, first run `uv lock`, then `uv sync --extra dev --locked`. Preserve an existing foreign lock rather than replacing it. Generating a local lock is dependency resolution, not proof that a dependency update or live runtime is ready.

`rag doctor`, `rag runtime-check`, `rag init`, ingestion, and real Ollama queries are operational checks that can inspect or change application state. They are separate from the fixture lane; use them only with deliberately selected disposable state and a configured local runtime. Do not ingest personal documents to validate source changes.

## Desktop lanes

The committed desktop lock uses Vite 8: use Node 20.19+ on the 20.x line, Node 22.12+ on the 22.x line, or Node 24+. These versions also satisfy the locked Vitest engine; Node 18 is insufficient. npm installs use `apps/desktop/package-lock.json`.

```bash
npm --prefix apps/desktop ci
npm --prefix apps/desktop test -- src/pages/LibraryPage.test.tsx
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
```

Choose an existing desktop test file for a focused change; `npm --prefix apps/desktop test -- --help` describes Vitest filters. The build runs TypeScript checking followed by Vite. There is no separate frontend lint script.

Rust/Tauri changes additionally require Rust, macOS/Xcode Command Line Tools for native desktop work, and `cargo test --locked --manifest-path apps/desktop/src-tauri/Cargo.toml`. Native packaging also uses `scripts/build_desktop_release.py` and its Python sidecar prerequisites; inspect that script before a separately authorized packaging run. A frontend build does not prove native packaging.

For changed UI behavior, run the relevant desktop tests and build, then use `npm --prefix apps/desktop run dev` for a localhost browser check with disposable fixture state. Do not connect that browser to a personal library or a real model merely to verify documentation. Tauri file dialogs and IPC require a separate native check; record unavailable browser/native lanes explicitly. Pure documentation changes do not require opening the application.

use serde::Serialize;
use std::env;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread::sleep;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

const API_HOST: &str = "127.0.0.1";
const API_BOOT_TIMEOUT: Duration = Duration::from_secs(5);
const TARGET_TRIPLE: &str = env!("TARGET_TRIPLE");

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionBootstrap {
    api_base_url: String,
    session_token: String,
    version: String,
    gpt_rag_home: String,
    runtime_mode: String,
    runtime_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendStatus {
    session_present: bool,
    api_alive: bool,
    worker_alive: bool,
    api_base_url: Option<String>,
    gpt_rag_home: String,
    runtime_mode: String,
    runtime_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeMode {
    Dev,
    Packaged,
}

impl RuntimeMode {
    fn as_str(&self) -> &'static str {
        match self {
            RuntimeMode::Dev => "dev",
            RuntimeMode::Packaged => "packaged",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeCommand {
    executable: PathBuf,
    args: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedRuntime {
    mode: RuntimeMode,
    source: String,
    api: RuntimeCommand,
    worker: RuntimeCommand,
}

struct BackendSession {
    bootstrap: SessionBootstrap,
    api_child: Child,
    worker_child: Child,
}

impl Drop for BackendSession {
    fn drop(&mut self) {
        stop_child(&mut self.api_child);
        stop_child(&mut self.worker_child);
    }
}

#[derive(Default)]
struct BackendState {
    session: Mutex<Option<BackendSession>>,
}

fn repo_root() -> Result<PathBuf, String> {
    repo_root_from_manifest_dir(Path::new(env!("CARGO_MANIFEST_DIR")))
}

fn repo_root_from_manifest_dir(manifest_dir: &Path) -> Result<PathBuf, String> {
    manifest_dir
        .join("../../..")
        .canonicalize()
        .map_err(|error| format!("Failed to resolve repo root: {error}"))
}

fn bundled_sidecar_paths(resource_dir: &Path, target_triple: &str) -> (PathBuf, PathBuf) {
    (
        resource_dir.join(format!("gpt-rag-api-{target_triple}")),
        resource_dir.join(format!("gpt-rag-worker-{target_triple}")),
    )
}

fn resolve_runtime_for(
    repo_root_path: &Path,
    resource_dir: Option<&Path>,
    override_path: Option<String>,
    target_triple: &str,
) -> Result<ResolvedRuntime, String> {
    if let Some(path) = override_path {
        let python = PathBuf::from(path);
        return Ok(ResolvedRuntime {
            mode: RuntimeMode::Dev,
            source: "python-override".to_string(),
            api: RuntimeCommand {
                executable: python.clone(),
                args: vec![
                    "-m".to_string(),
                    "gpt_rag.gui_api".to_string(),
                    "--host".to_string(),
                    API_HOST.to_string(),
                ],
            },
            worker: RuntimeCommand {
                executable: python,
                args: vec!["-m".to_string(), "gpt_rag.gui_worker".to_string()],
            },
        });
    }

    let candidate = repo_root_path.join(".venv/bin/python");
    if candidate.exists() {
        return Ok(ResolvedRuntime {
            mode: RuntimeMode::Dev,
            source: "repo-venv".to_string(),
            api: RuntimeCommand {
                executable: candidate.clone(),
                args: vec![
                    "-m".to_string(),
                    "gpt_rag.gui_api".to_string(),
                    "--host".to_string(),
                    API_HOST.to_string(),
                ],
            },
            worker: RuntimeCommand {
                executable: candidate,
                args: vec!["-m".to_string(), "gpt_rag.gui_worker".to_string()],
            },
        });
    }

    if let Some(resources) = resource_dir {
        let (api_sidecar, worker_sidecar) = bundled_sidecar_paths(resources, target_triple);
        if api_sidecar.exists() && worker_sidecar.exists() {
            return Ok(ResolvedRuntime {
                mode: RuntimeMode::Packaged,
                source: "bundled-sidecar".to_string(),
                api: RuntimeCommand {
                    executable: api_sidecar,
                    args: vec!["--host".to_string(), API_HOST.to_string()],
                },
                worker: RuntimeCommand {
                    executable: worker_sidecar,
                    args: Vec::new(),
                },
            });
        }
    }

    Err(
        "Could not resolve a desktop backend runtime. Set GPT_RAG_GUI_PYTHON for development \
or build bundled sidecars for packaged use."
            .to_string(),
    )
}

fn resolve_runtime(app: &AppHandle) -> Result<ResolvedRuntime, String> {
    let resources = app.path().resource_dir().ok();
    resolve_runtime_for(
        &repo_root()?,
        resources.as_deref(),
        env::var("GPT_RAG_GUI_PYTHON").ok(),
        TARGET_TRIPLE,
    )
}

fn reserve_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind((API_HOST, 0))
        .map_err(|error| format!("Failed to reserve a local API port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("Failed to read local API port: {error}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn spawn_backend_process(
    command_spec: &RuntimeCommand,
    extra_args: &[String],
    session_token: &str,
    port: u16,
    gpt_rag_home: &str,
) -> Result<Child, String> {
    let mut command = Command::new(&command_spec.executable);
    command
        .args(&command_spec.args)
        .args(extra_args)
        .env("GPT_RAG_GUI_TOKEN", session_token)
        .env("GPT_RAG_GUI_PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if !gpt_rag_home.is_empty() {
        command.env("GPT_RAG_HOME", gpt_rag_home);
    }

    command.spawn().map_err(|error| {
        format!(
            "Failed to start backend sidecar {:?}: {error}",
            command_spec.executable
        )
    })
}

fn wait_for_api(port: u16) -> Result<(), String> {
    let start = Instant::now();
    while start.elapsed() < API_BOOT_TIMEOUT {
        if TcpStream::connect((API_HOST, port)).is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(120));
    }
    Err("Timed out waiting for the local GUI API to start.".to_string())
}

fn stop_child(child: &mut Child) {
    match child.try_wait() {
        Ok(Some(_)) => {}
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
        }
        Err(_) => {
            let _ = child.kill();
        }
    }
}

fn child_is_alive(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(None))
}

fn session_liveness(session: &mut BackendSession) -> (bool, bool) {
    (
        child_is_alive(&mut session.api_child),
        child_is_alive(&mut session.worker_child),
    )
}

fn should_reuse_session(session: &mut BackendSession) -> bool {
    let (api_alive, worker_alive) = session_liveness(session);
    api_alive && worker_alive
}

fn shutdown_session(slot: &mut Option<BackendSession>) {
    let _ = slot.take();
}

#[cfg_attr(not(test), allow(dead_code))]
fn clear_stale_session(slot: &mut Option<BackendSession>) -> bool {
    let should_clear = slot
        .as_mut()
        .map(|session| !should_reuse_session(session))
        .unwrap_or(false);
    if should_clear {
        shutdown_session(slot);
        return true;
    }
    false
}

fn backend_status_for(
    slot: &mut Option<BackendSession>,
    gpt_rag_home: String,
    runtime_mode: String,
    runtime_source: String,
) -> BackendStatus {
    if let Some(session) = slot.as_mut() {
        let (api_alive, worker_alive) = session_liveness(session);
        return BackendStatus {
            session_present: true,
            api_alive,
            worker_alive,
            api_base_url: Some(session.bootstrap.api_base_url.clone()),
            gpt_rag_home,
            runtime_mode: session.bootstrap.runtime_mode.clone(),
            runtime_source: session.bootstrap.runtime_source.clone(),
        };
    }
    BackendStatus {
        session_present: false,
        api_alive: false,
        worker_alive: false,
        api_base_url: None,
        gpt_rag_home,
        runtime_mode,
        runtime_source,
    }
}

fn current_home_dir() -> String {
    env::var("GPT_RAG_HOME").unwrap_or_default()
}

fn start_backend_session(app: &AppHandle) -> Result<BackendSession, String> {
    let runtime = resolve_runtime(app)?;
    let port = reserve_loopback_port()?;
    let token = Uuid::new_v4().to_string();
    let home_dir = current_home_dir();
    let version = app.package_info().version.to_string();

    let mut api_child = spawn_backend_process(
        &runtime.api,
        &[String::from("--port"), port.to_string()],
        &token,
        port,
        &home_dir,
    )?;

    if let Err(error) = wait_for_api(port) {
        stop_child(&mut api_child);
        return Err(error);
    }

    let worker_child = match spawn_backend_process(&runtime.worker, &[], &token, port, &home_dir) {
        Ok(child) => child,
        Err(error) => {
            stop_child(&mut api_child);
            return Err(error);
        }
    };

    Ok(BackendSession {
        bootstrap: SessionBootstrap {
            api_base_url: format!("http://{API_HOST}:{port}"),
            session_token: token,
            version,
            gpt_rag_home: home_dir,
            runtime_mode: runtime.mode.as_str().to_string(),
            runtime_source: runtime.source,
        },
        api_child,
        worker_child,
    })
}

fn ensure_session(
    app: &AppHandle,
    slot: &mut Option<BackendSession>,
    force_restart: bool,
) -> Result<SessionBootstrap, String> {
    if force_restart {
        shutdown_session(slot);
    } else if let Some(session) = slot.as_mut() {
        if should_reuse_session(session) {
            return Ok(session.bootstrap.clone());
        }
        shutdown_session(slot);
    }

    let session = start_backend_session(app)?;
    let bootstrap = session.bootstrap.clone();
    *slot = Some(session);
    Ok(bootstrap)
}

#[tauri::command]
fn bootstrap_session(
    app: AppHandle,
    state: State<BackendState>,
) -> Result<SessionBootstrap, String> {
    let mut guard = state
        .session
        .lock()
        .map_err(|_| "Failed to lock backend session state.".to_string())?;
    let result = ensure_session(&app, &mut guard, false);
    if let Err(error) = &result {
        eprintln!("desktop bootstrap_session failed: {error}");
    }
    result
}

#[tauri::command]
fn restart_session(app: AppHandle, state: State<BackendState>) -> Result<SessionBootstrap, String> {
    let mut guard = state
        .session
        .lock()
        .map_err(|_| "Failed to lock backend session state.".to_string())?;
    let result = ensure_session(&app, &mut guard, true);
    if let Err(error) = &result {
        eprintln!("desktop restart_session failed: {error}");
    }
    result
}

#[tauri::command]
fn backend_status(app: AppHandle, state: State<BackendState>) -> Result<BackendStatus, String> {
    let mut guard = state
        .session
        .lock()
        .map_err(|_| "Failed to lock backend session state.".to_string())?;
    let runtime = resolve_runtime(&app)?;
    Ok(backend_status_for(
        &mut guard,
        current_home_dir(),
        runtime.mode.as_str().to_string(),
        runtime.source,
    ))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendState::default())
        .invoke_handler(tauri::generate_handler![
            bootstrap_session,
            restart_session,
            backend_status
        ])
        .setup(|app| {
            app.emit("desktop-ready", true)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running GPT_RAG desktop");
}

#[cfg(test)]
mod tests {
    use super::{
        backend_status_for, bundled_sidecar_paths, clear_stale_session, repo_root_from_manifest_dir,
        resolve_runtime_for, BackendSession, RuntimeMode, SessionBootstrap, TARGET_TRIPLE,
    };
    use std::env;
    use std::fs;
    use std::process::{Command, Stdio};
    use std::thread::sleep;
    use std::time::Duration;
    use uuid::Uuid;

    fn spawn_child(command: &str) -> std::process::Child {
        Command::new("sh")
            .args(["-c", command])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn child")
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let path = env::temp_dir().join(format!("gpt-rag-desktop-{label}-{}", Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    fn session_with_children(
        api_child: std::process::Child,
        worker_child: std::process::Child,
        runtime_mode: &str,
        runtime_source: &str,
    ) -> BackendSession {
        BackendSession {
            bootstrap: SessionBootstrap {
                api_base_url: "http://127.0.0.1:8787".to_string(),
                session_token: "token".to_string(),
                version: "0.1.0".to_string(),
                gpt_rag_home: "/tmp/test-home".to_string(),
                runtime_mode: runtime_mode.to_string(),
                runtime_source: runtime_source.to_string(),
            },
            api_child,
            worker_child,
        }
    }

    #[test]
    fn resolve_runtime_prefers_python_override() {
        let repo_root = temp_dir("override");
        let resolved = resolve_runtime_for(
            &repo_root,
            None,
            Some("/tmp/custom-python".into()),
            TARGET_TRIPLE,
        )
        .expect("override should resolve");

        assert_eq!(resolved.mode, RuntimeMode::Dev);
        assert_eq!(resolved.source, "python-override");
        assert_eq!(
            resolved.api.executable,
            std::path::PathBuf::from("/tmp/custom-python")
        );
    }

    #[test]
    fn resolve_runtime_prefers_bundled_sidecars_when_present() {
        let repo_root = temp_dir("bundled");
        let resources = temp_dir("resources");
        let (api_sidecar, worker_sidecar) = bundled_sidecar_paths(&resources, TARGET_TRIPLE);
        fs::write(&api_sidecar, "binary").expect("write api");
        fs::write(&worker_sidecar, "binary").expect("write worker");

        let resolved = resolve_runtime_for(&repo_root, Some(&resources), None, TARGET_TRIPLE)
            .expect("bundled sidecars should resolve");

        assert_eq!(resolved.mode, RuntimeMode::Packaged);
        assert_eq!(resolved.source, "bundled-sidecar");
        assert_eq!(resolved.api.executable, api_sidecar);
        assert_eq!(resolved.worker.executable, worker_sidecar);
    }

    #[test]
    fn resolve_runtime_uses_repo_venv_when_present() {
        let repo_root = temp_dir("repo-venv");
        let python = repo_root.join(".venv/bin/python");
        fs::create_dir_all(python.parent().expect("missing parent")).expect("mkdir");
        fs::write(&python, "#!/usr/bin/env python3\n").expect("write python stub");

        let resolved =
            resolve_runtime_for(&repo_root, None, None, TARGET_TRIPLE).expect("repo venv");

        assert_eq!(resolved.mode, RuntimeMode::Dev);
        assert_eq!(resolved.source, "repo-venv");
        assert_eq!(resolved.api.executable, python);
    }

    #[test]
    fn repo_root_from_manifest_dir_walks_back_to_project_root() {
        let root = temp_dir("project-root");
        let manifest_dir = root.join("apps/desktop/src-tauri");
        fs::create_dir_all(&manifest_dir).expect("mkdir manifest dir");
        let canonical_root = root.canonicalize().expect("canonical root");

        let resolved =
            repo_root_from_manifest_dir(&manifest_dir).expect("project root should resolve");

        assert_eq!(resolved, canonical_root);
    }

    #[test]
    fn clear_stale_session_drops_dead_children() {
        let alive = spawn_child("sleep 2");
        let exited = spawn_child("exit 0");
        sleep(Duration::from_millis(50));

        let mut slot = Some(session_with_children(alive, exited, "dev", "repo-venv"));
        assert!(clear_stale_session(&mut slot));
        assert!(slot.is_none());
    }

    #[test]
    fn backend_status_reports_liveness_and_runtime_metadata() {
        let api_child = spawn_child("sleep 2");
        let worker_child = spawn_child("sleep 2");
        let mut slot = Some(session_with_children(
            api_child,
            worker_child,
            "packaged",
            "bundled-sidecar",
        ));

        let status = backend_status_for(
            &mut slot,
            "/tmp/status-home".to_string(),
            "packaged".to_string(),
            "bundled-sidecar".to_string(),
        );
        assert!(status.session_present);
        assert!(status.api_alive);
        assert!(status.worker_alive);
        assert_eq!(
            status.api_base_url.as_deref(),
            Some("http://127.0.0.1:8787")
        );
        assert_eq!(status.gpt_rag_home, "/tmp/status-home");
        assert_eq!(status.runtime_mode, "packaged");
        assert_eq!(status.runtime_source, "bundled-sidecar");
    }
}

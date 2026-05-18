from pathlib import Path
from trainer_and_simulator_functions import launch_inference_sim

def batch_inference(
        run_dir: Path,
        unity_env_path: Path,
        patched_yaml_path: Path,
        out_root: Path,
        episodes: int,
        timeout_s: int = 300,
        seed: int = 17,
        run_ids: list[str] | None = None,
        behaviour_name: str = "OctagonAgentSolo"
):
    '''
    Run inference sequentially on all trained models in run_dir/results/.
    If run_ids is provided, only those run IDs are used.
    Otherwise, all subdirectories under run_dir/results/ are treated as run IDs.
    '''
    run_dir = Path(run_dir)
    results_root = run_dir / "results"

    if run_ids is None:
        run_ids = sorted([
            d.name for d in results_root.iterdir() if d.is_dir()
        ])

    if not run_ids:
        raise FileNotFoundError(f"No run directories found in {results_root}")

    print(f"Found {len(run_ids)} models to run inference on: {run_ids}")

    logs = {}
    for i, run_id in enumerate(run_ids):
        print(f"\n=== [{i+1}/{len(run_ids)}] Running inference: {run_id} ===")
        out_path = Path(out_root) / run_id
        try:
            log = launch_inference_sim(
                run_dir=run_dir,
                unity_env_path=unity_env_path,
                patched_yaml_path=patched_yaml_path,
                train_run_id=run_id,
                behavior_name=behaviour_name,
                out_path=out_path,
                episodes=episodes,
                timeout_s=timeout_s,
                seed=seed,
            )
            logs[run_id] = log
            print(f"=== Finished: {run_id} -> {log} ===")
        except Exception as e:
            print(f"=== FAILED: {run_id} — {e} ===")
            logs[run_id] = None

    return logs

# Helper function to find the latest .onnx file for a given run ID
# Currently unused
def find_latest_onnx(run_dir: Path, run_id: str) -> Path:
    candidates = list((run_dir /run_id / "OctagonAgentSolo").rglob("*.onnx"))
    
    if not candidates:
        raise FileNotFoundError(f"No .onnx found for {run_id}")
    
    return max(candidates, key=lambda p: p.stat().st_mtime)
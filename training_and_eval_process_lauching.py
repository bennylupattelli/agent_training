import numpy as np
import torch
from pathlib import Path
import shutil
import subprocess, time, os, signal


def launch_training(
        patched_yaml: Path,
        unity_env_path: Path,
        run_id: str,
        torch_device: str,
        num_envs: int,
        base_port: int,
        seed: int | None = None,
        extra_args: list[str] | None = None,
        cwd: Path | None = None,
):
    '''
    3) Launch unity mlagents-learn for one training run.
    REMEBER TO ACTIVATE THE ENV WITH mlagents.
    Assumes that the patched yaml file includes gamma.
    Unity will read the action cost from run_config_path.
    '''

    patched_yaml = Path(patched_yaml)
    unity_env_path = Path(unity_env_path)

    if not patched_yaml.exists():
        raise FileNotFoundError(f"Patched yaml file not found: {patched_yaml}")
    if not unity_env_path.exists():
        raise FileNotFoundError(f"Unity environment not found: {unity_env_path}")

    # build the command-line invocation
    cmd = [
        #"xvfb-run",
        #"-a",
        #"-s",
        #"-screen 0 1280x1024x24",
        "mlagents-learn", # executable
        str(patched_yaml), # path to the yaml config
        "--env", str(unity_env_path), # points to the compiled unity environment
        "--torch-device", torch_device, # specify torch device (e.g., cuda:0)
        "--num-envs", str(num_envs), # number of parallel environments
        "--no-graphics", # headless mode
        "--run-id", run_id, # specify run id
        "--base-port", str(base_port),
        "--timeout-wait", "300",
        "--force",
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # this allows passing extra arguments, e.g., --num-envs=4
    if extra_args:
        cmd.extend(extra_args)

    # Popen allows us to monitor the output of the process in real-time, which is useful for debugging and logging.
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd else None
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end='')  # print each line as it is received
        p.wait()  # wait for the process to complete
        if p.returncode != 0: 
            raise subprocess.CalledProcessError(p.returncode, cmd)


def _start_process(cmd, log_file: Path, cwd: Path | None = None):
    '''
    Start subprocess to allow termination on both Windows and Mac/Linux systems.
    '''
    popen_kwargs = {
        "stdout": open(log_file, "w"),
        "stderr": subprocess.STDOUT,
        "text": True,
        "cwd": str(cwd) if cwd else None,
    }

    if os.name == "nt":
        # Windows: create a new process group
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Mac/Linux: create a new session so we can kill the whole process group
        popen_kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **popen_kwargs)

def _terminate_process_tree(p: subprocess.Popen, force: bool = False):
    """
    Terminate the subprocess cleanly, using OS-appropriate logic.
    force=False -> polite termination
    force=True  -> hard kill
    """
    if p.poll() is not None:
        return  # already exited

    if os.name == "nt":
        # Windows does not support os.killpg / os.getpgid.
        # terminate() is polite, kill() is forceful.
        if force:
            p.kill()
        else:
            p.terminate()
    else:
        pgid = os.getpgid(p.pid)
        if force:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.killpg(pgid, signal.SIGTERM)




def run_eval(cmd, out_path: Path, poll_s=0.2, timeout_s=300, cwd: Path | None = None):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Signal file written by Unity / simulation code when evaluation is complete
    done_file = out_path / "DONE.txt"
    if done_file.exists():
        done_file.unlink()

    log_file = out_path / "mlagents_stdout.log"

    p = None
    log_handle = None

    try:
        log_handle = open(log_file, "w")
        popen_kwargs = {
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
            "cwd": str(cwd) if cwd else None,
        }

        if os.name == "nt":
            # Windows: new process group
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Linux/macOS: new session/process group
            popen_kwargs["start_new_session"] = True

        p = subprocess.Popen(cmd, **popen_kwargs)

        t0 = time.time()

        while True:
            if done_file.exists():
                break

            if p.poll() is not None:
                raise RuntimeError(
                    f"mlagents-learn exited early with code {p.returncode}. See {log_file}"
                )

            if time.time() - t0 > timeout_s:
                _terminate_process_tree(p, force=False)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(p, force=True)
                    p.wait()

                raise TimeoutError(f"Timed out waiting for DONE.txt. See {log_file}")

            time.sleep(poll_s)

        # DONE.txt appeared: shut process down cleanly
        if p.poll() is None:
            _terminate_process_tree(p, force=False)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(p, force=True)
                p.wait()

    finally:
        if log_handle is not None and not log_handle.closed:
            log_handle.close()

    return log_file




def launch_inference_sim(run_dir: Path,
                unity_env_path: Path,
                patched_yaml_path: Path,
                train_run_id: str,
                out_path: Path,
                episodes: int,
                base_port: int,
                timeout_s: int = 5000, # more time is needed for more than 100 episodes
                seed: int | None = None,
                ):
    '''3) Launch unity mlagents-learn in inference mode for simulations.'''

    run_dir = Path(run_dir)
    unity_env_path = Path(unity_env_path)
    patched_yaml_path = Path(patched_yaml_path)
    out_path = Path(out_path)

    if not patched_yaml_path.exists():
        raise FileNotFoundError(f"patched_yaml_path not found")
    if not unity_env_path.exists():
        raise FileNotFoundError(f"unity_env_path not found")
        
    results_dir = run_dir / "results" / train_run_id
    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    # create the output directory for this simulation run, where the DONE.txt file will be printed.
    out_path.mkdir(parents=True, exist_ok=True)

    # build the command-line invocation for inference mode
    cmd = [
        "mlagents-learn",
        str(patched_yaml_path),
        "--run-id", train_run_id,          
        "--resume",
        "--inference",
        "--base-port", str(base_port),
        "--env", str(unity_env_path),
        "--no-graphics",                 
        "--env-args",
        "--sim_out", str(out_path.resolve()),
        "--sim_eps", str(episodes),
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(cmd)
    
    # to actually launch the process
    return run_eval(cmd, out_path, poll_s=0.2, timeout_s=timeout_s, cwd=run_dir)

import random

import numpy as np
import torch
from pathlib import Path
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
import shutil
import subprocess, time, os, signal

# Shared training seeds, read off the 260914 run dirs (each run's configuration.yaml records the
# seed it was trained with). sequential_runs uses these by default so every batch trains the same
# seeds in the same order, which makes a comparison between two batches paired rather than
# unpaired: a difference between arms is then the hyperparameter, not the luck of the draw.
# Index i is the seed for model i+1, so 260929_0007 trains on the same seed as 260914_0007.
# Pass seeds=None to sequential_runs to draw fresh random ones instead.
# Seed 31 is the one 260914_0031 was started on. Seeds 32-100 were drawn with
# random.Random(20261007).randint(0, 10**6), skipping duplicates, and are fixed from here on.
DEFAULT_TRAINING_SEEDS = [
    726131, 448213, 155319,  68621, 584835, 878205, 915294, 970535, 410034, 163489,
    631403, 378759, 201455, 885226, 310413, 863655, 376274, 216613, 497442, 462529,
    392983, 598083, 755089,   8859, 152318, 507504, 171882, 333555, 728747, 308611,
    781674, 239536, 855470, 866465, 371902,  79975, 305416, 638504, 652686, 737937,
    852944, 591615, 782112, 106299, 293665, 906875, 854934, 134590, 524966, 582848,
    863391, 788275, 106756, 861424, 108205, 497665, 927080,   6008, 662738, 199707,
    959135, 757672, 125589, 568331, 174203, 184612, 115273, 330918, 728814, 206647,
    380192, 629395, 506881, 935593, 245983,   2789, 591375, 354064, 278096, 776574,
    572113, 690624,  49626, 223829, 573282,  60869,  50766, 763569,  91368,  41869,
     17666,  38461, 535873, 972007,  27638, 476591, 120245, 670209, 896769, 302873,
]

'''
Step 1) Sample parameters from the prior
Step 2) Patch yaml for agents to include new parameter values
Step 3) Launch training session with new parameters
Step 4) Run simulations

Note: For Linux use the unity_env_path needs to point to the .x86_64 file (the build)
      set unity_env_path = Path("/your_path/to/env.x86_64").
      If the path is incorrect, you will get an error.
      Also, make sure you have mlagents installed and activate the correct conda environment if using conda.    
'''


def sample_first_thetas(
        N: int,
        gamma_range=(0.95, 0.9999),
        sp_range=(2.5e-6, 4e-3),
        device="cpu",
):
    '''
    1) Sample N thetas from the prior distribution. 
    '''
    g0, g1 = gamma_range
    p0, p1 = sp_range

    gamma = torch.rand(N, 1, device=device) * (g1 - g0) + g0
    sp = torch.rand(N, 1, device=device) * (p1 - p0) + p0

    # sample uniformly in log-space for sp 
    #log_sp = torch.rand(N, 1, device=device) * (torch.log(torch.tensor(p1)) - torch.log(torch.tensor(p0))) + torch.log(torch.tensor(p0))
    
    # exponentiate back to normal space
    #sp = torch.exp(log_sp)

    # stack parameters into theta
    theta = torch.cat([gamma, sp], dim=1)

    theta = theta.tolist()
    theta = np.array(theta)

    return theta



def patch_agents_yaml(
        template_yaml: str | Path,
        output_yaml: str | Path,
        run_dir: str | Path,
        gamma: float = 0.99,
        step_penalty: float = 1e-2,
        behaviour_name: str | None = None,
        extrinsic_reward_key: str = "extrinsic",
) -> None:
    
    '''
    2) Load yaml file for agents and patch gamma.
    template_yaml: path to yaml file,
    run_dir: directory to save the patched yaml file,
    gamma: PPO discount,
    step_penalty: penalty for each step taken,
    behaviour_name: name of the behaviour to patch, if None patch all behaviours,
    extrinsic_reward_key: key in the yaml file for extrinsic reward,
    '''

    '''
    template_yaml = "/Path/to/your/original.yaml", e.g., template_yaml = "/Users/benny/Repos/octagon/Assets/Scripts/MLConfigFiles/nEnvsSoloConfig.yaml"
    
    run_dir = "/Path/to/your/new/agents/folder/in/run/directory", e.g., run_dir = Path("runs") / "run_0001"
    
    output_yaml = run_dir / "SoloConfig.yaml", e.g., output_yaml = run_dir / "SoloConfig.yaml"
    '''

    template_yaml = Path(template_yaml)
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True) # creates a directory at the specified path, including any necessary parent directories.

    with template_yaml.open("r") as f:
        cfg = yaml.load(f) # loads the yaml file
    
    if "behaviors" not in cfg or not isinstance(cfg["behaviors"], dict) or "environment_parameters" not in cfg:
        raise KeyError("Invalid yaml file format: 'behaviors' or 'environment_parameters' keys not found or 'behaviors' is not a dictionary.")
        # checks if the yaml file has the correct format

    behaviours = cfg["behaviors"]
    env_parameters = cfg["environment_parameters"]

    # select which behaviours to patch
    target_behaviours = [behaviour_name] if behaviour_name else list(behaviours.keys())
    missing = [b for b in target_behaviours if b not in behaviours]
    if missing:
        raise KeyError(f"Behaviours not found in yaml file: {missing}. Found: {list(behaviours.keys())}")
    
    if "step_penalty" not in env_parameters:
        raise KeyError("Missing environment parameter 'step_penalty'")

    for b in target_behaviours:
        bcfg = behaviours[b]

        # patch gamma and instrinsic reward strength
        reward_signals = bcfg.get("reward_signals")
        if reward_signals is None or extrinsic_reward_key not in reward_signals:
            raise KeyError(
                f"Missing reward signal '{extrinsic_reward_key}' not found in behaviour '{b}'."
                f"Available reward signals: {list(reward_signals.keys()) if isinstance(reward_signals, dict) else reward_signals}"
            )
        
        extrinsic_cfg = reward_signals[extrinsic_reward_key]

        if not isinstance(extrinsic_cfg, dict):
            raise KeyError(f"Invalid reward signal configuration format in behaviour '{b}'.")
        
        extrinsic_cfg["gamma"] = float(gamma)

    for p in env_parameters:
        if p == "step_penalty":
            env_parameters[p] = float(step_penalty)
            
    with output_yaml.open("w") as f:
        yaml.dump(cfg, f)




def launch_training(
        patched_yaml: Path,
        unity_env_path: Path,
        run_id: str,
        torch_device: str,
        num_envs: int,
        base_port: int,
        seed: int | None = None,
        extra_args: list[str] | None = None,
        run_dir: Path | None = None,
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
        "mlagents-learn", # executable
        str(patched_yaml), # path to the yaml config
        "--env", str(unity_env_path), # points to the compiled unity environment
        "--torch-device", torch_device, # specify torch device (e.g., cuda:0)
        "--num-envs", str(num_envs), # number of parallel environments
        "--no-graphics", # headless mode
        "--run-id", run_id, # specify run id
        "--base-port", str(base_port),
        
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])
        print(f"Using seed: {seed} for training run {run_id}")

    # this allows passing extra arguments, e.g., --num-envs=4
    if extra_args:
        cmd.extend(extra_args)

    # Popen allows us to monitor the output of the process in real-time, which is useful for debugging and logging.
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(run_dir) if run_dir else None
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



# Note that running inference can also take a seed. I'm just going to default set this to 17 for now 
def launch_inference_sim(run_dir: Path,
                unity_env_path: Path,
                patched_yaml_path: Path,
                train_run_id: str,
                out_path: Path,
                episodes: int,
                base_port: int,
                timeout_s: int = 5000, # more time is needed for more than 100 episodes
                seed: int | None = 17,
                deterministic: bool = False,
                trial_seq: Path | str | None = None,
                ):
    '''3) Launch unity mlagents-learn in inference mode for simulations.
    deterministic: if True, pass --deterministic so the policy selects greedy
    actions (dist.argmax). Defaults to False everywhere (solo/social batch
    inference AND the tournament) so actions are SAMPLED from the categorical
    policy, matching how the models behaved during training. Greedy is retained
    only as an opt-in for cases where run-to-run reproducibility is needed.
    trial_seq: optional path to a predetermined trial-sequence JSON (see
    trial_sequences/generate_trial_sequence.py). When given, the Unity build
    replays that exact string of trials and stops once it is exhausted, so every
    model sees identical trials. When None, the build generates trials randomly.'''

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
        # --deterministic must precede --env-args (which consumes all remaining args)
        *(["--deterministic"] if deterministic else []),
        "--env", str(unity_env_path),
        "--no-graphics",                 
        "--env-args",
        "--sim_out", str(out_path.resolve()),
        "--sim_eps", str(episodes),
    ]

    # Predetermined trial sequence (optional). Passed through --env-args so the
    # Unity build reads it via GetCommandLineArgs(); absence => random trials.
    if trial_seq is not None:
        trial_seq = Path(trial_seq).resolve()
        if not trial_seq.exists():
            raise FileNotFoundError(f"trial_seq file not found: {trial_seq}")
        cmd.extend(["--trial_seq", str(trial_seq)])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(cmd)
    
    # to actually launch the process
    return run_eval(cmd, out_path, poll_s=0.2, timeout_s=timeout_s, cwd=run_dir)




def sequential_runs(
        in_yaml: Path,
        run_dir: Path,
        yaml_injection: bool = False,
        gamma: float = 0.99,
        sp: float = 1e-2,
        behaviour_name: str = "OctagonAgentSolo",
        unity_build: Path = Path("/Users/benny/Builds/OctagonAgentSolo.app"),
        # IMPORTANT:
        # for Linux use the unity_build path that points to the .x86_64 file (the build), e.g. unity_build = Path("/path/to/env.x86_64")
        # for Windows use the unity_build path that points to the .exe file (the build), e.g. unity_build = Path("C:/path/to/env.exe")
        base_run_id: str = "run",
        device: str = "cpu",
        n_models: int = 1,
        run_id_offset: int = 0,
        simulate: bool = False,
        n_envs: int = 1,
        n_steps: int = 5,
        # Passed through to mlagents-learn, but ONLY for the first model in the
        # batch (i == 0). Flags like --resume/--initialize-from are meant for the
        # run being continued, not for the fresh runs that follow it; applying
        # them to every model would make each subsequent run try to resume its
        # own (non-existent) prior checkpoint.
        extra_args: list[str] | None = None,
        # Training seeds, taken by absolute model number so every batch trains the same seeds in
        # the same order. Pass None to draw fresh random seeds, which makes this batch
        # incomparable pairwise with any other.
        seeds: list[int] | None = DEFAULT_TRAINING_SEEDS,
):
    '''
    in_yaml = "/Path/to/original.yaml"
    run_dir = "/Path/to/your/new/agents/run/directory", e.g., run_dir = Path("runs") assuming that the current working directory is the sbi directory.
    unity_build = Path("/Path/to/your/unity/build"), e.g. unity_build=Path("/Users/benny/Builds/OctagonAgentSolo.app")
    base_run_id = prefix to every simulation run id
    device = torch device for training (e.g., "cpu" or "cuda:0")
    n_models = number of agents (models) to train sequentially
    n_envs = number of parallel environments to use for training (e.g., 1, 2, 4, etc.)
    n_eps = number of episodes to run for each simulation in the inference step (e.g., 1000, 10000, 100000, etc.)
    '''
    
    run_dir = Path(run_dir)
    in_yaml = Path(in_yaml)
    # take the shared seeds by absolute model number, so model 0007 of any batch trains on the
    # same seed as model 0007 of every other batch. seeds=None draws fresh random ones instead
    if seeds is None:
        seeds = [random.randint(0, 1e6) for _ in range(n_models)]
        print(f"Using FRESH RANDOM seeds: {seeds} for {n_models} sequential runs")
        print("NB batches trained on different seeds cannot be compared pairwise")
    else:
        need = run_id_offset + n_models
        if len(seeds) < need:
            raise ValueError(
                f"need {need} seeds for {n_models} models at offset {run_id_offset}, "
                f"but only {len(seeds)} were given. Extend the list or pass seeds=None")
        seeds = list(seeds)[run_id_offset:need]
        print(f"Using shared seeds: {seeds} for {n_models} sequential runs "
              f"(models {run_id_offset + 1:04d}-{need:04d})")

    # run_id_offset is the number of models already finished, so the first model trained here is
    # run_id_offset + 1. To resume a model in progress, set run_id_offset to its number minus one
    first_run_id = f"{base_run_id}_{run_id_offset + 1:04d}"
    if extra_args and "--resume" in extra_args and not (run_dir / "results" / first_run_id).exists():
        raise FileNotFoundError(
            f"--resume was passed but results/{first_run_id} does not exist. run_id_offset counts "
            f"finished models, so to resume model N set run_id_offset=N-1")

    for i in range(n_models):
        seed = seeds[i]
        run_id = f"{base_run_id}_{run_id_offset + 1 + i:04d}" # create a unique run ID for each simulation run, e.g. "sbi_solo_run_0001", "sbi_solo_run_0002", etc.
        # patched_yaml_path = run_dir / f"Config_{run_id}.yaml" # create a unique patched yaml file for each run, e.g. "SoloConfig_0001.yaml", "SoloConfig_0002.yaml", etc.
        yaml_filename = in_yaml.name # get the filename from the input yaml path, e.g. "SoloConfig.yaml"
        patched_yaml_path = run_dir / yaml_filename # create a unique patched yaml file for each run
        train_port = 5005 + 20 * i
        sim_port   = 5015 + 20 * i  

        if yaml_injection:
            # this function replaces the placeholders in the yaml file with the sampled parameters
            patch_agents_yaml(
                template_yaml=in_yaml,
                output_yaml=patched_yaml_path,
                gamma=gamma,
                step_penalty=sp,
                behaviour_name=behaviour_name,
                extrinsic_reward_key="extrinsic",
            )
        else:
            patched_yaml_path = in_yaml # if yaml_injection is False, use the original yaml file for all runs without patching

        print(f"patched yaml for run {run_id} with gamma={gamma} and step_penalty={sp} to {patched_yaml_path}")

        print(f"launching training for run {run_id} with config {patched_yaml_path}")
        if extra_args:
            if i == 0:
                print(f"  applying extra_args to first model only: {extra_args}")
            else:
                print(f"  skipping extra_args for {run_id} (only the first model gets them)")
        # this function launches one training run with the specified yaml file and Unity environment build
        launch_training(
            patched_yaml=patched_yaml_path,
            unity_env_path=unity_build,
            run_id=run_id,
            torch_device=device,
            num_envs=n_envs,
            base_port=train_port,
            seed=seed,
            run_dir=run_dir,
            # Only the first model in the batch receives the extra args (e.g. --resume);
            # every later run is a fresh model and must start clean.
            extra_args=(extra_args if i == 0 else None),
        )

        if simulate == True:
            print(f"launching inference for run {run_id}")
            # this function launches one inference run using the trained model from the training run
            # specify the number of steps to run 
            # the random seed is not currently implemented in the inference code, but it is included here for future use
            try:
                launch_inference_sim(
                    run_dir=run_dir,
                    unity_env_path=unity_build,
                    patched_yaml_path=patched_yaml_path.resolve(),
                    train_run_id=run_id,
                    out_path=run_dir / "simulations" / f"sim_{run_id}",
                    episodes=n_eps,
                    base_port=sim_port,
                    seed=seed,
                )
                
            except TimeoutError as e:
                print(f"[WARNING] Simulation timed out for {run_id}: {e}")
                print("[WARNING] Continuing to next model.")
            except Exception as e:
                print(f"[WARNING] Simulation failed for {run_id}: {type(e).__name__}: {e}")
                print("[WARNING] Continuing to next model.")

        time.sleep(5)




def sbi_simulator(
        n: int,
        in_yaml: Path,
        run_dir: Path,
        work_dir: Path,
        behaviour_name: str = "OctagonAgentSolo",
        unity_build: Path = Path("/Users/benny/Builds/OctagonAgentSolo.app"),
        # IMPORTANT: for Linux use the unity_build path that points to the .x86_64 file (the build), e.g. unity_build = Path("/path/to/env.x86_64")
        base_run_id: str = "sbi_solo_run",
        device: str = "cpu",
        simulate: bool = False,
        n_envs: int = 1,
        n_eps: int = 5,
        seed: int | None = None,
):
    '''Umbrella function to run the whole pipeline with one command:
    1. Sample N batches of parameters from the prior distribution
    2. For each batch of parameters, patch the template yaml file and launch a training run with the patched yaml and Unity environment build
    3. After each training run, launch an inference run using the trained model from the training run, specifying the number of episodes to run and the random seed for future use (currently not implemented in the inference code)'''
    
    run_dir = Path(run_dir)
    in_yaml = Path(in_yaml)
    work_dir = Path(work_dir)

    thetas = sample_first_thetas(n) # output is (n, 2) np array

    #print(f"Sampled thetas:\n{thetas}")

    # get N batches of parameter values from the prior distribution
    for i in range(n):
        _, sp = map(float, thetas[i]) # convert tensor values to floats for yaml patching

        run_id = f"{base_run_id}_{i:04d}" # create a unique run ID for each simulation run, e.g. "sbi_solo_run_0001", "sbi_solo_run_0002", etc.
        patched_yaml_path = run_dir / f"SoloConfig_{run_id}.yaml" # create a unique patched yaml file for each run, e.g. "SoloConfig_0001.yaml", "SoloConfig_0002.yaml", etc.

        train_port = 5005 + 20 * i
        sim_port   = 5015 + 20 * i

        # this function replaces the placeholders in the yaml file with the sampled parameters
        patch_agents_yaml(
            template_yaml=in_yaml,
            output_yaml=patched_yaml_path,
            gamma=0.99,
            step_penalty=sp,
            behaviour_name=behaviour_name,
            extrinsic_reward_key="extrinsic"
        )
    
        # this function launches one training run with the specified yaml file and Unity environment build
        launch_training(
            patched_yaml=patched_yaml_path,
            unity_env_path=unity_build,
            run_id=run_id,
            torch_device=device,
            num_envs=n_envs,
            base_port=train_port,
            seed=seed,
            run_dir=run_dir,
        )

        if simulate == True:
            # this function launches one inference run using the trained model from the training run
            # specify the number of episodes to run 
            # the random seed is not currently implemented in the inference code, but it is included here for future use
            try:
                launch_inference_sim(
                    run_dir=run_dir,
                    unity_env_path=unity_build,
                    patched_yaml_path=patched_yaml_path.resolve(),
                    train_run_id=run_id,
                    out_path=work_dir / "simulations" / f"sim_{run_id}",
                    episodes=n_eps,
                    base_port=sim_port,
                    seed=seed,
                )

            except TimeoutError as e:
                print(f"[WARNING] Simulation timed out for {run_id}: {e}")
                print("[WARNING] Continuing to next model.")
            except Exception as e:
                print(f"[WARNING] Simulation failed for {run_id}: {type(e).__name__}: {e}")
                print("[WARNING] Continuing to next model.")

        time.sleep(5)


def write_extended_config(
        template_yaml: Path,
        out_yaml: Path,
        max_steps: float,
        behaviour_name: str | None = None,
) -> Path:
    '''
    Write a copy of a training config with max_steps raised, for extending finished runs.

    Takes the config the runs were trained with, the path to write, and the new max_steps.
    Only max_steps changes, so the extension continues the same training.

    This writes a separate file instead of editing the template, because the template is
    shared with future fresh batches and raising its max_steps would change what a standard
    run means for every batch after this one.
    '''

    template_yaml, out_yaml = Path(template_yaml), Path(out_yaml)
    with template_yaml.open("r") as f:
        cfg = yaml.load(f)

    behaviours = cfg.get("behaviors")
    if not behaviours:
        raise KeyError(f"No 'behaviors' block in {template_yaml}")
    targets = [behaviour_name] if behaviour_name else list(behaviours)
    missing = [b for b in targets if b not in behaviours]
    if missing:
        raise KeyError(f"Behaviours not in {template_yaml}: {missing}. Found: {list(behaviours)}")

    for b in targets:
        old = behaviours[b].get("max_steps")
        if old is not None and float(old) >= float(max_steps):
            raise ValueError(
                f"max_steps for '{b}' is already {float(old):.0f}, not below the requested "
                f"{float(max_steps):.0f}. mlagents would exit on --resume with nothing left "
                "to train."
            )
        behaviours[b]["max_steps"] = float(max_steps)

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.dump(cfg, f)
    return out_yaml


def checkpoint_step(model_run_dir: Path, behaviour_name: str) -> int:
    '''
    Return the global step a finished run reached. Takes the run dir and behaviour name.
    Used to confirm a run sits at the old ceiling before extending it, and to report how
    far each run got.
    '''

    import torch
    ckpt = Path(model_run_dir) / behaviour_name / "checkpoint.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return int(state["global_step"]["_GlobalSteps__global_step"].item())


def runs_below_max_steps(results_root, max_steps, behaviour_name="OctagonAgentSocial",
                         exclude=(), verbose=True):
    """
    Return the run ids under results_root whose checkpoint is below max_steps.

    Takes the folder of runs and the target step count. Use it to pick what still needs
    extending: resume_training_runs raises on a run already at the ceiling, so a batch that
    mixes finished and unfinished runs has to be filtered first. Runs with no checkpoint are
    reported and left out.
    """
    results_root = Path(results_root)
    exclude = set(exclude)
    todo, done, broken = [], [], []
    for d in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if d.name in exclude:
            continue
        if not (d / behaviour_name / "checkpoint.pt").exists():
            broken.append(d.name)
            continue
        step = checkpoint_step(d, behaviour_name)
        (todo if step < max_steps else done).append((d.name, step))

    if verbose:
        print(f"{len(todo)} runs below {max_steps:.0f} steps, {len(done)} already at or past it")
        if broken:
            print(f"  no {behaviour_name}/checkpoint.pt: {broken}")
        if todo:
            gains = [max_steps - s for _, s in todo]
            print(f"  steps to add: {min(gains):,.0f} to {max(gains):,.0f} each, "
                  f"{sum(gains):,.0f} total")
    return [name for name, _ in todo]


def resume_training_runs(
        run_ids: list[str],
        results_root: Path,
        run_dir: Path,
        in_yaml: Path,
        unity_build: Path,
        max_steps: float,
        behaviour_name: str = "OctagonAgentSocial",
        device: str = "cuda:0",
        n_envs: int = 16,
        base_port: int = 5005,
        port_step: int = 20,
        seed: int | None = 17,
        extended_yaml: Path | None = None,
        dry_run: bool = False,
):
    '''
    Extend a set of ALREADY-TRAINED runs to a higher max_steps, one after another.

    Takes the run ids to extend, the folder holding them, the mlagents working dir (the
    parent of `results/`), the config they were trained with, the build, and the new
    max_steps. Returns {run_id: reached_step}.

    Use this rather than sequential_runs. That function trains a contiguous block of fresh
    models, so it passes extra_args (--resume) to the first model only and every later run
    starts clean. Here every run already exists and needs resuming, and the ids are a
    hand-picked set rather than a range.

    Two things have to line up, or mlagents does nothing and reports success:
      - max_steps in the config must exceed the step the runs stopped at, or the resumed run
        exits with nothing left to train.
      - the run id must locate the run under <run_dir>/results/. mlagents resolves output as
        os.path.join(results_dir, run_id), so a run filed into a subfolder needs that
        subfolder in its id, such as "2607/260723/260723_0034".
    Both are checked before anything launches, so a mistake costs a message rather than a
    night of no-op runs.
    '''

    results_root, run_dir = Path(results_root), Path(run_dir)
    in_yaml, unity_build = Path(in_yaml), Path(unity_build)

    if not unity_build.exists():
        raise FileNotFoundError(f"Unity build not found: {unity_build}")

    # mlagents writes to <run_dir>/results/<run_id>, so the id carries any subfolders the
    # runs were filed into. Derive that prefix rather than trusting the bare name.
    try:
        prefix = results_root.relative_to(run_dir / "results")
    except ValueError as e:
        raise ValueError(
            f"results_root ({results_root}) must sit under run_dir/results "
            f"({run_dir / 'results'}). mlagents resolves output as results_dir/run_id."
        ) from e

    # every run must exist and still be below the new ceiling
    starts = {}
    for rid in run_ids:
        d = results_root / rid
        if not (d / behaviour_name / "checkpoint.pt").exists():
            raise FileNotFoundError(f"No {behaviour_name}/checkpoint.pt under {d}")
        starts[rid] = checkpoint_step(d, behaviour_name)
        if starts[rid] >= max_steps:
            raise ValueError(
                f"{rid} is already at {starts[rid]} steps, at or past max_steps "
                f"{max_steps:.0f}. It would resume and exit with nothing to train."
            )

    extended_yaml = Path(extended_yaml) if extended_yaml else \
        in_yaml.with_name(f"{in_yaml.stem}_{max_steps:.0e}.yaml".replace("+0", "e").replace("ee", "e"))
    write_extended_config(in_yaml, extended_yaml, max_steps, behaviour_name)
    print(f"extended config written: {extended_yaml}  (max_steps -> {max_steps:.0f})")

    print(f"\nResuming {len(run_ids)} runs under {results_root}")
    for rid in run_ids:
        print(f"  {rid}: {starts[rid]:>9,} -> {max_steps:>11,.0f}  "
              f"(+{max_steps - starts[rid]:,.0f} steps)  run-id '{prefix / rid}'")
    if dry_run:
        print("\ndry_run=True, nothing launched.")
        return starts

    reached = {}
    for i, rid in enumerate(run_ids):
        mlagents_run_id = str(prefix / rid)
        print(f"\n=== [{i+1}/{len(run_ids)}] Resuming {rid} "
              f"(from {starts[rid]:,} steps) as run-id '{mlagents_run_id}' ===")
        try:
            launch_training(
                patched_yaml=extended_yaml,
                unity_env_path=unity_build,
                run_id=mlagents_run_id,
                torch_device=device,
                num_envs=n_envs,
                base_port=base_port + port_step * i,
                seed=seed,
                run_dir=run_dir,
                extra_args=["--resume"],      # every run is resumed, unlike sequential_runs
            )
            reached[rid] = checkpoint_step(results_root / rid, behaviour_name)
            print(f"=== Finished {rid}: {starts[rid]:,} -> {reached[rid]:,} steps ===")
        except Exception as e:
            reached[rid] = None
            print(f"=== Failed {rid}. {type(e).__name__}: {e} ===")

    print("\nSummary:")
    for rid in run_ids:
        got = reached.get(rid)
        print(f"  {rid}: {starts[rid]:,} -> {f'{got:,}' if got else 'FAILED'}")
    return reached

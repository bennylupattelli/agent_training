import numpy as np
import torch
from pathlib import Path
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
import shutil
import subprocess, time, os, signal

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
        sp_range=(1e-4, 1e-3),
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

def sample_split_cost(
        #ratio_range,
        move_range=(0.00025, 0.004),
        turn_range=(0.00025, 0.004),
        n = 25,
        distribution="even_steps",
        seed=None,
):
    '''
    (not implemented: Ratio range determines the relative size of each parameter with respect to the other.)
    n is the number of samples for each parameter which will result in a n*n total samples.
    Move and turn ranges determine the absolute lower and upper ceilings for each parameter.
    Distribution argument determines how the parameter values are sampled:
        at even ratio steps,
        randomly via uniform distribution,
        randomly via log distribution
    '''

    rng = np.random.default_rng(seed)

    #r0, r1 = ratio_range
    m0, m1 = move_range
    t0, t1 = turn_range

    if m0 <= 0 or m1 <= 0 or t0 <= 0 or t1 <= 0:
        raise ValueError("Cost ranges must be positive.")

    if m0 >= m1 or t0 >= t1:
        raise ValueError("Each range must be ordered as (lower, upper).")

    if distribution=="even_steps":

        n_per_dim = int(np.sqrt(n))

        if n_per_dim ** 2 != n:
            raise ValueError(
                "For even steps, n must be a perfect square"
            )

        translation_costs = np.linspace(m0, m1, n_per_dim)
        turning_costs = np.linspace(t0, t1, n_per_dim)
        
        theta = np.array([
        [tc,rc]
        for tc in translation_costs
        for rc in turning_costs
        ])
    
    elif distribution=="uniform":

        translation_costs = rng.uniform(m0, m1, n)
        turning_costs = rng.uniform(t0, t1, n)

        theta = np.column_stack([
            translation_costs,
            turning_costs
        ])
    
    elif distribution=="log_uniform":

        translation_costs = np.exp(
            rng.uniform(np.log(m0), np.log(m1), n)
        )

        turning_costs = np.exp(
            rng.uniform(np.log(t0), np.log(t1), n)
        )

        theta = np.column_stack([
            translation_costs,
            turning_costs
        ])
    
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: even_steps, uniform, log_uniform"
        )

    return theta


def patch_agents_yaml(
        template_yaml: str | Path,
        output_yaml: str | Path,
        split_cost: bool,
        translation_cost: float,
        turning_cost: float,
        gamma: float = 0.99,
        step_penalty: float = 1e-2,
        behaviour_name: str | None = None,
        extrinsic_reward_key: str = "extrinsic",
) -> None:
    
    '''
    2) Load yaml file for agents and patch gamma.
    template_yaml: path to yaml file,
    output_yaml: path to write the patched yaml file,
    translation_cost: penalty for each translational movement step,
    turning_cost: penalty for each rotational movement step,
    gamma: PPO discount,
    step_penalty: penalty for each step taken,
    behaviour_name: name of the behaviour to patch, if None patch all behaviours,
    extrinsic_reward_key: key in the yaml file for extrinsic reward,
    '''

    '''
    template_yaml = "/Path/to/your/original.yaml", e.g., template_yaml = "/Users/benny/Repos/octagon/Assets/Scripts/MLConfigFiles/nEnvsSoloConfig.yaml"
    
    run_dir = "/Path/to/your/new/agents/folder/in/run/directory", e.g., run_dir = Path("runs") / "run_0001"
    
    output_yaml = run_dir/name_of_new_yaml_file.yaml, e.g., output_yaml = run_dir / "SoloConfig.yaml"
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
    
    if not split_cost:
        if "step_penalty" not in env_parameters:
            raise KeyError("Missing environment parameter 'step_penalty'")
    if split_cost:
        if "translation_cost" not in env_parameters:
            raise KeyError("Missing environment parameter 'translation_cost'")   

        if "turning_cost" not in env_parameters:
            raise KeyError("Missing environment parameter 'turning_cost'")   
    
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
        if not split_cost:
            if p == "step_penalty":
                env_parameters[p] = float(step_penalty)
        if split_cost:
            if p == "translation_cost":
                env_parameters[p] = float(translation_cost)
            if p == "turning_cost":
                env_parameters[p] = float(turning_cost)          
      
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
        "mlagents-learn", # executable
        str(patched_yaml), # path to the yaml config
        "--env", str(unity_env_path), # points to the compiled unity environment
        "--torch-device", torch_device, # specify torch device (e.g., cuda:0)
        "--num-envs", str(num_envs), # number of parallel environments
        "--no-graphics", # headless mode
        "--run-id", run_id, # specify run id
        "--base-port", str(base_port),
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




def sequential_runs(
        in_yaml: Path,
        run_dir: Path,
        work_dir: Path,
        gamma: float = 0.99,
        sp: float = 1e-2,
        behaviour_name: str = "OctagonAgentSolo",
        unity_build: Path = Path("/Users/benny/Builds/OctagonAgentSolo.app"),
        # IMPORTANT:
        # for Linux use the unity_build path that points to the .x86_64 file (the build), e.g. unity_build = Path("/path/to/env.x86_64")
        # for Windows use the unity_build path that points to the .exe file (the build), e.g. unity_build = Path("C:/path/to/env.exe")
        base_run_id: str = "run",
        device: str = "cpu",
        n_agents: int = 1,
        simulate: bool = False,
        n_envs: int = 1,
        n_eps: int = 5,
        seed: int | None = None,
):
    '''
    in_yaml = "/Path/to/original.yaml"
    run_dir = "/Path/to/your/new/agents/run/directory", e.g., run_dir = Path("runs") assuming that the current working directory is the sbi directory.
    work_dir = "/Path/to/your/work/directory/for/simulations", e.g., work_dir = Path("/Users/benny/Documents/swc/bayesian-inference/sbi")
    unity_build = Path("/Path/to/your/unity/build"), e.g. unity_build=Path("/Users/benny/Builds/OctagonAgentSolo.app")
    base_run_id = prefix to every simulation run id
    device = torch device for training (e.g., "cpu" or "cuda:0")
    n_agents = number of agents (models) to train sequentially
    n_envs = number of parallel environments to use for training (e.g., 1, 2, 4, etc.)
    n_eps = number of episodes to run for each simulation in the inference step (e.g., 1000, 10000, 100000, etc.)
    '''
    
    run_dir = Path(run_dir)
    in_yaml = Path(in_yaml)
    work_dir = Path(work_dir)

    for i in range(n_agents):
        
        run_id = f"{base_run_id}_{i:04d}" # create a unique run ID for each simulation run, e.g. "sbi_solo_run_0001", "sbi_solo_run_0002", etc.
        patched_yaml_path = run_dir / f"Config_{run_id}.yaml" # create a unique patched yaml file for each run, e.g. "SoloConfig_0001.yaml", "SoloConfig_0002.yaml", etc.
        
        train_port = 5005 + 20 * i
        sim_port   = 5015 + 20 * i  

        # this function replaces the placeholders in the yaml file with the sampled parameters
        patch_agents_yaml(
            template_yaml=in_yaml,
            output_yaml=patched_yaml_path,
            gamma=gamma,
            step_penalty=sp,
            behaviour_name=behaviour_name,
            extrinsic_reward_key="extrinsic",
        )

        print(f"patched yaml for run {run_id} with gamma={gamma} and step_penalty={sp} to {patched_yaml_path}")

        print(f"launching training for run {run_id} with config {patched_yaml_path}")
        # this function launches one training run with the specified yaml file and Unity environment build
        launch_training(
            patched_yaml=patched_yaml_path,
            unity_env_path=unity_build,
            run_id=run_id,
            torch_device=device,
            num_envs=n_envs,
            base_port=train_port,
            seed=seed,
            cwd=work_dir,
        )

        if simulate == True:
            print(f"launching inference for run {run_id}")
            # this function launches one inference run using the trained model from the training run
            # specify the number of episodes to run 
            # the random seed is not currently implemented in the inference code, but it is included here for future use
            try:
                launch_inference_sim(
                    run_dir=work_dir,
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


def sbi_simulator(
        n: int,
        in_yaml: Path,
        #run_dir: Path,
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
        step_penalty=False,
        split_penalty=False,
        input_split_theta=None
):
    '''Umbrella function to run the whole pipeline with one command:
    1. Sample N batches of parameters from the prior distribution
    2. For each batch of parameters, patch the template yaml file and launch a training run with the patched yaml and Unity environment build
    3. After each training run, launch an inference run using the trained model from the training run, specifying the number of episodes to run and the random seed for future use (currently not implemented in the inference code)'''
    
    #run_dir = Path(run_dir)
    work_dir = Path(work_dir).resolve()
    in_yaml = Path(in_yaml).resolve()
    unity_build = Path(unity_build).resolve()

    config_dir = work_dir / "configs"
    #results_dir = work_dir / "results"
    simulations_dir = work_dir / "simulations"
    #logs_dir = work_dir / "logs"

    for d in [config_dir, simulations_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if step_penalty and split_penalty:
        raise ValueError("Choose only one: step_penalty or split_penalty.")
    
    if step_penalty:
        thetas = sample_first_thetas(n) # output is (n, 2) np array
    elif split_penalty and input_split_theta is None:
        thetas = sample_split_cost(n=n, seed=seed)
    elif split_penalty and input_split_theta is not None:
        thetas = input_split_theta
    else:
        raise ValueError(
            f"Desired parameter not given. Step penalty is '{step_penalty}', split penalty is '{split_penalty}'"
            "Choose from: step_penalty=True or step_penalty=False"
        )
    #print(f"Sampled thetas:\n{thetas}")

    # get N batches of parameter values from the prior distribution
    for i, theta in enumerate(thetas):

        run_id = f"{base_run_id}_{i:04d}" # create a unique run ID for each simulation run, e.g. "sbi_solo_run_0001", "sbi_solo_run_0002", etc.
        
        patched_yaml_path = config_dir / f"SoloConfig_{run_id}.yaml" # create a unique patched yaml file for each run, e.g. "SoloConfig_0001.yaml", "SoloConfig_0002.yaml", etc.

        train_port = 5005 + 20 * i
        sim_port   = 5015 + 20 * i

        if step_penalty:
            _, sp = map(float, theta) # convert tensor values to floats for yaml patching

            # this function replaces the placeholders in the yaml file with the sampled parameters
            patch_agents_yaml(
                template_yaml=in_yaml,
                output_yaml=patched_yaml_path,
                split_cost=False,
                gamma=0.99,
                step_penalty=sp,
                behaviour_name=behaviour_name,
                extrinsic_reward_key="extrinsic"
            )

        elif split_penalty:
            tc, rc = map(float, theta) 
        
            # this function replaces the placeholders in the yaml file with the sampled parameters
            patch_agents_yaml(
                template_yaml=in_yaml,
                output_yaml=patched_yaml_path,
                split_cost=True,
                translation_cost=tc,
                turning_cost=rc,
                gamma=0.99,
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
            cwd=work_dir,
        )

        if simulate:
            # this function launches one inference run using the trained model from the training run
            # specify the number of episodes to run 
            # the random seed is not currently implemented in the inference code, but it is included here for future use
            try:
                launch_inference_sim(
                    run_dir=work_dir,
                    unity_env_path=unity_build,
                    patched_yaml_path=patched_yaml_path,
                    train_run_id=run_id,
                    out_path=simulations_dir/f"sim_{run_id}",
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

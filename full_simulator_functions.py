import numpy as np
from pathlib import Path
import subprocess, time, os, signal
from parameter_sampling import sample_first_thetas, sample_split_cost
from config_file_patching import patch_agents_yaml
from training_and_eval_process_lauching import launch_training, launch_inference_sim



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
        split_penalty=False
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
    elif split_penalty:
        thetas = sample_split_cost(n=n, seed=seed)
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

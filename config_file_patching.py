import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True


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
from pathlib import Path
import trainer_and_simulator_functions as trainer


in_yaml = Path("/ceph/scratch/bgenca/octagon/template_yaml/SplitCostSolo.yaml")
work_dir = Path("/ceph/scratch/bgenca/octagon/agent_training/agent_training")
unity_env_path = Path("/ceph/scratch/bgenca/octagon/builds/OctagonAgentSoloLinux/SoloOctagon.x86_64")
step_penalty = False
split_penalty = True
behaviour_name = "OctagonAgentSolo"
n = 1
seed = 7
base_run_id = "split_cost_hpc"


trainer.sbi_simulator(
    n = 1,
    in_yaml = in_yaml,
    work_dir = work_dir,
    unity_build = unity_env_path,
    base_run_id = "split_cost_hpc",
    device = "gpu",
    simulate = True,
    n_eps = 1000,
    n_envs = 1,
    seed=seed,
    split_penalty=split_penalty
)
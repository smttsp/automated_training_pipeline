import wandb


def create_wandb_project(project_name, runtime_str):
    wandb.init(
        project=project_name,
        name=f"run_{runtime_str}"
    )
    return wandb

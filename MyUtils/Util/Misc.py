import hydra
from omegaconf import OmegaConf
import os
def is_sweep():
    """
    Check if the current run is part of a WandB sweep.
    """
    if wandb.run is None:
        return False
    return wandb.run.sweep_id is not None
def load_and_override_config(config_dir, config_name, manual_overrides={}):
    """
    Load configuration with Hydra, manually override parameters, and integrate with WandB.

    Args:
    - config_dir (str): Directory path where configuration files are stored.
    - config_name (str): Name of the configuration file to load without the extension.
    - manual_overrides (dict): Dictionary of parameters to override manually.

    Returns:
    - OmegaConf.DictConfig: The final configuration object after all overrides.
    """

    # Initialize Hydra and load the base configuration
    # hydra.initialize(config_path=config_dir)
    # cfg = hydra.compose(config_name + ".yaml")
    cfg = OmegaConf.load(f"{config_dir}/{config_name}.yaml")
    
    # Apply manual overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.create(manual_overrides))
    
    
    # Check if running under WandB and apply WandB configuration if it's a sweep
    if wandb.run is not None:
        # Assuming wandb has been initialized outside this function in your main workflow
        wandb_config = wandb.config
        print("wandb_config",wandb_config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb_config)))

    cfg.is_sweep= is_sweep()
    print("cfg",cfg)
    return cfg


import wandb

def get_or_create_sweep_id(project_name, sweep_config,force_create=False):
    """
    Get or create a sweep ID for the given project.

    This function checks if there is a file named '{project_name}_sweep_id.txt' that contains the sweep ID.
    If the file exists, it reads the sweep ID from the file.
    If the file does not exist, it creates a new sweep and writes the sweep ID to the file.

    Args:
    project_name (str): The name of the project.
    sweep_config (dict): The configuration of the sweep.

    Returns:
    str: The sweep ID.
    """
    sweep_id_folder = 'sweep_ids'
    sweep_id_file = f'{project_name}_sweep_id.txt'
    sweep_id_file = os.path.join(sweep_id_folder, sweep_id_file)
    if force_create:
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        with open(sweep_id_file, 'w') as file:
            file.write(sweep_id)
        return sweep_id
    # Check if the sweep ID file exists
    if os.path.exists(sweep_id_file):
        # If the file exists, read the sweep ID from the file
        with open(sweep_id_file, 'r') as file:
            sweep_id = file.read().strip()
    else:
        # If the file does not exist, create a new sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        # Make sure the directory exists
        os.makedirs(sweep_id_folder, exist_ok=True)
        # Write the sweep ID to the file
        with open(sweep_id_file, 'w') as file:
            file.write(sweep_id)
    
    return sweep_id
def cleanup_and_save_top_models(project_name, username, sweep_id, top_x, sort_metric="epoch/val_acc", artifact_name="best-model", delete_other=False, local_save_path="./best_models"):
    """
    Identifies the top X best runs from a Weights & Biases sweep,
    deletes artifacts from the other runs, saves the top models locally for evaluation, and saves the best model overall as `overall_best_model`.

    Args:
        project_name (str): The name of the wandb project.
        username (str): Your wandb username.
        sweep_id (str): The sweep ID containing the runs.
        top_x (int): The number of best runs to retain.
        sort_metric (str): The metric name to use for sorting the best runs.
        artifact_name (str): The name of the model artifact to save or delete.
        delete_other (bool): Whether to delete artifacts that aren't in the top X.
        local_save_path (str): Path where the top models will be saved locally.
    """
    # Initialize the wandb API
    api = wandb.Api()

    # Construct the project path
    project_path = f"{username}/{project_name}"

    # Fetch all runs associated with the specified project and sweep
    runs = api.runs(path=project_path, filters={"sweep": sweep_id})

    # Sort runs by the specified metric, defaulting to 0 if the metric isn't found
    sorted_runs = sorted(runs, key=lambda run: run.summary.get(sort_metric, 0), reverse=True)
    print(f"Found {len(sorted_runs)} runs in the sweep.")

    # Identify the top X runs
    top_runs = sorted_runs[:top_x]
    print(f"Identified the top {top_x} runs.")

    # Get the best overall run
    best_overall_run = sorted_runs[0]
    print(f"Best overall run: {best_overall_run.name}")

    # Create a set of run IDs to keep
    top_run_ids = {run.id for run in top_runs}
    print(f"Top run IDs: {top_run_ids}")

    # Create a directory to save the top models locally
    os.makedirs(local_save_path, exist_ok=True)

    # Process each run and decide whether to save or delete its artifact
    for run in sorted_runs:
        try:
            # Find the list of artifacts associated with the current run
            artifacts = list(run.logged_artifacts())

            # Find the artifact that matches the specified artifact_name
            artifact = next((a for a in artifacts if artifact_name in a.name), None)

            if artifact is None:
                raise ValueError(f"No artifact named {artifact_name} found for run {run.name}")

            if run.id in top_run_ids:
                # Download and save the model locally if it's in the top X
                artifact_dir = artifact.download()
                local_model_path = os.path.join(local_save_path, f"{run.name}.keras")
                os.rename(os.path.join(artifact_dir, "best_model.keras"), local_model_path)
                print(f"Saved {local_model_path} locally.")

                # Save the overall best model as `overall_best_model.keras`
                if run == best_overall_run:
                    overall_best_path = os.path.join(local_save_path, "overall_best_model.keras")
                    os.rename(local_model_path, overall_best_path)
                    print(f"Saved the best overall model as {overall_best_path}.")
            else:
                # Delete the artifact if it's not in the top X
                if delete_other:
                    artifact.delete()
                    print(f"Deleted artifact from run {run.name}.")
                else:
                    print(f"Skipping deletion of artifact from run {run.name}.")
        except Exception as e:
            print(f"Could not process artifact for run {run.name}: {e}")

    print("Completed processing the models.")

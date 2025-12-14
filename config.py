import os
from pathlib import Path

# Base directory for local models
# Base directory for local models
LOCAL_MODELS_DIR = Path("models")

# Manual .env parsing
def load_env_file():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()
USE_OFFLINE = os.getenv("useOffline", "false").lower() == "true"

def get_local_model_path(repo_id: str, filename: str) -> str | None:
    """
    Returns the absolute path to a local model file if it exists.
    Expected structure: models/Organization/RepoName/filename
    Example: models/Zyphra/Zonos-v0.1-transformer/config.json
    """
    local_path = LOCAL_MODELS_DIR / repo_id / filename
    if local_path.exists():
        return str(local_path.resolve())
    return None

def has_local_model(repo_id: str) -> bool:
    """Checks if the essential files for a model exist locally."""
    # We assume at least config.json and model.safetensors are needed for the main model
    # For speaker embedding, we might need different checks, but this is a good baseline.
    config_path = get_local_model_path(repo_id, "config.json")
    model_path = get_local_model_path(repo_id, "model.safetensors")
    
    # Simple check: if we found both, return True.
    # Note: This logic is primarily for the main Zonos model.
    if config_path and model_path:
        return True
    
    return False

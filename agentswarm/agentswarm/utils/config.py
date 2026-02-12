"""Configuration utilities for AgentSwarm.

This module provides functions for loading and saving configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a file.
    
    Supports YAML and JSON formats.
    
    Args:
        path: Path to the configuration file.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["swarm"]["name"])
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    
    if suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
        return yaml.safe_load(content)
    elif suffix == ".json":
        import json
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def save_config(
    config: Dict[str, Any], 
    path: Union[str, Path]
) -> None:
    """Save configuration to a file.
    
    Supports YAML and JSON formats.
    
    Args:
        config: Configuration dictionary.
        path: Path to save the configuration file.
    
    Raises:
        ValueError: If the file format is not supported.
    
    Example:
        >>> config = {"swarm": {"name": "my-swarm"}}
        >>> save_config(config, "config.yaml")
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("PyYAML is required to save YAML files. Install with: pip install pyyaml")
        content = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    elif suffix == ".json":
        import json
        content = json.dumps(config, indent=2)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")
    
    path.write_text(content, encoding="utf-8")


def merge_configs(
    base: Dict[str, Any], 
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    The override config takes precedence over the base config.
    
    Args:
        base: Base configuration.
        override: Override configuration.
    
    Returns:
        Merged configuration.
    
    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}}
        >>> merged = merge_configs(base, override)
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

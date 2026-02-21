import json
from typing import Dict, Any


class ConfigManager:
    """Centralized configuration management."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def get_data_config(script_dir: str) -> Dict[str, Any]:
        """Get data configuration relative to script directory."""
        import os
        config_path = os.path.join(script_dir, '../config.json')
        return ConfigManager.load_config(config_path)

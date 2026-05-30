import yaml
import argparse
from typing import Dict, Any

class ConfigNode:
    """A helper node that allows attribute-style dot access for dictionaries."""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            elif isinstance(value, list):
                setattr(self, key, [ConfigNode(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in value]
            else:
                result[key] = value
        return result

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return str(self.__dict__)


def load_config(config_path: str) -> ConfigNode:
    """Loads a YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return ConfigNode(data)


def parse_args_and_config(default_config_path: str = None) -> ConfigNode:
    """Parses command line arguments and merges them with a YAML configuration file."""
    parser = argparse.ArgumentParser(description="DoTPFN Clinical Predictive Pipeline")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to YAML config file")
    parser.add_argument("--device", type=str, help="Compute device override (e.g. 'cpu', 'cuda')")
    parser.add_argument("--epochs", type=int, help="Number of training epochs override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--batch_size", type=int, help="Batch size override")
    parser.add_argument("--model_save_dir", type=str, help="Model save directory override")
    
    args, unknown = parser.parse_known_args()
    
    if args.config is None:
        raise ValueError("Must provide a config file path via --config.")
        
    config = load_config(args.config)
    
    # Apply overrides
    if args.device:
        config.device = args.device
    if args.epochs:
        if hasattr(config, 'training'):
            config.training.epochs = args.epochs
        else:
            config.epochs = args.epochs
    if args.lr:
        if hasattr(config, 'training'):
            config.training.lr = args.lr
        else:
            config.lr = args.lr
    if args.batch_size:
        if hasattr(config, 'training'):
            config.training.batch_size = args.batch_size
        else:
            config.batch_size = args.batch_size
    if args.model_save_dir:
        if hasattr(config, 'training'):
            config.training.model_save_dir = args.model_save_dir
        else:
            config.model_save_dir = args.model_save_dir
            
    return config

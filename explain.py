import sys
import os

# Guarantee local src/ package resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotpfn.utils.config import parse_args_and_config
from dotpfn.scripts.explain import run_explanation

if __name__ == "__main__":
    config = parse_args_and_config()
    run_explanation(config)

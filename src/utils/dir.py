"""Build the output path/filename for saved inference results.

`generate_output_path` has a sensible default (model name + #prompts + intervention), 
so the pipeline runs as-is. Customize it to encode more of your run's parameters 
into the filename so different runs don't overwrite each other.
"""

import argparse
from pathlib import Path


def generate_output_path(args: argparse.Namespace) -> str:
    """Generate the output path for a run's results.

    If `args.output` is set, it is honored directly. Otherwise, a filename is built
    from the run parameters in `args`.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The output path for the run's results.
    """
    if args.output is not None:
        return args.output

    # Build filename from run parameters to prevent overwriting different runs
    model_name = str(args.model_path).replace("/", "--")
    intervention = getattr(args, "intervention", None) is not None
    num_prompts = getattr(args, "num_prompts", "unknown")
    
    # Capture additional parameters for better run differentiation
    seed = getattr(args, "seed", None)
    batch_size = getattr(args, "batch_size", None)
    learning_rate = getattr(args, "learning_rate", None)
    
    # Build components list with all relevant parameters
    components = [
        f"m={model_name}",
        f"p={num_prompts}",
        f"int={intervention}",
    ]
    
    # Add optional parameters if they exist
    if seed is not None:
        components.append(f"seed={seed}")
    if batch_size is not None:
        components.append(f"bs={batch_size}")
    if learning_rate is not None:
        components.append(f"lr={learning_rate}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    filename = "_".join(components) + ".pt"
    return str(output_dir / filename)
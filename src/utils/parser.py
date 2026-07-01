"""Command-line arguments for the inference / activation-extraction entry point (src/main.py).

The generic run arguments (model, layers, output, ...) are provided as-is. Add any
task-specific arguments your prompts or analysis need in the marked TODO section;
a small worked example is kept below as a commented illustration.
"""

import argparse
from typing import List


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add inference command-line arguments to the parser.

    Args:
        parser: The argument parser to add arguments to.
    """
    # --- Generic run arguments (useful for most analysis runs) ---
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="HuggingFace repo or local path to the model.",
    )
    parser.add_argument(
        "--layers",
        "-l",
        nargs="+",
        type=int,
        required=False,
        help="List of layer indices to analyze. Defaults to all decoder layers.",
    )
    parser.add_argument(
        "--num-prompts",
        "-p",
        default=1000,
        type=int,
        help="Number of prompts to evaluate on. Defaults to 1000.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "-mnt",
        type=int,
        required=False,
        default=200,
        help="Maximum number of new tokens to generate. Defaults to 200.",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        help="Directory where the output file will be saved.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )

    # --- Core mechinterp toggles (capture + intervention) ---
    # The flags are generic; their specifics are defined by your inference code:
    #   - what activations / token positions --capture-geometry records, and
    #   - the file format the --intervention spec is read from.
    parser.add_argument(
        "--capture-geometry",
        action="store_true",
        help="Capture detailed per-position activations (MLP neurons, attention heads, residual stream) "
        "in addition to the model output. Omit for lightweight runs such as large ablation sweeps.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Path to an intervention spec (e.g. a JSON listing neurons/heads to ablate or patch). "
        "When provided, an intervention pass is run after the baseline.",
    )

    # ----------------------------------------------------------------------- #
    # TODO: add task-specific arguments for your study here.
    # ----------------------------------------------------------------------- #
    #
    # The following arguments support an integer addition task where operands are in
    # the range 0..999 (up to 3 digits) and prompts can be generated in multiple languages.
    #
    # These arguments are intended to be forwarded from src/main.py into
    # whatever prompt/dataset generator you implement (e.g. PromptDataset.generate_prompts).
    #
    # ----------------------------------------------------------------------- #
    task_group = parser.add_argument_group(
        title="Task-specific arguments",
        description="Arguments that control prompt generation and task parameters for the addition study.",
    )

    task_group.add_argument(
        "--few-shot-examples",
        "-fs",
        type=int,
        default=4,
        help="Number of solved (few-shot) examples to show before each question. Defaults to 4.",
    )

    task_group.add_argument(
        "--max-digits",
        "-md",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Maximum number of digits per operand (1..3). "
            "1 -> 0..9, 2 -> 0..99, 3 -> 0..999. Defaults to 3."
        ),
    )

    task_group.add_argument(
        "--min-number",
        "-minn",
        type=int,
        default=0,
        help="Minimum integer value (inclusive) to use for operands. Defaults to 0.",
    )
    task_group.add_argument(
        "--max-number",
        "-maxn",
        type=int,
        default=999,
        help="Maximum integer value (inclusive) to use for operands. Defaults to 999.",
    )

    task_group.add_argument(
        "--languages",
        "-lg",
        nargs="+",
        choices=["english", "hindi", "mandarin", "all"],
        default=["english"],
        help=(
            "One or more languages to generate prompts in. Choices: english, hindi, mandarin, all. "
            "Use 'all' to select all supported languages. Defaults to ['english']."
        ),
    )

    task_group.add_argument(
        "--write-numbers-as-words",
        "-w",
        action="store_true",
        help=(
            "If set, prompts will write the operands as words rather than digits where possible "
            "(e.g. 'seven plus five' vs '7+5'). Language-dependent formatting should be handled "
            "in your prompt generator."
        ),
    )

    task_group.add_argument(
        "--prompt-template",
        "-pt",
        type=str,
        choices=["compact", "verbose"],
        default="compact",
        help=(
            "Template style for prompts. 'compact' produces short prompts like '7+5=', "
            "'verbose' produces more natural-language prompts like 'What is 7 plus 5?'. "
            "Your prompt generator should implement these variants."
        ),
    )

    # Helpful note: the parser only sets/collects flags. Validate relations after parsing
    # in src/main.py (for example, ensure min-number <= max-number and clamp by max-digits).
    #
    # Example (in src/main.py after args = parser.parse_args()):
    #
    #     # Ensure numeric bounds are consistent and respect max_digits:
    #     max_by_digits = 10**args.max_digits - 1
    #     args.max_number = min(args.max_number, max_by_digits)
    #     if args.min_number < 0:
    #         args.min_number = 0
    #     if args.max_number < args.min_number:
    #         raise ValueError("max_number must be >= min_number")
    #
    #     # Expand 'all' languages to the full list:
    #     if "all" in args.languages:
    #         args.languages = ["english", "hindi", "mandarin"]
    #
    # Then forward args to your prompt/dataset generator:
    #     prompts = PromptDataset.generate_prompts(
    #         num_prompts=args.num_prompts,
    #         few_shot=args.few_shot_examples,
    #         min_value=args.min_number,
    #         max_value=args.max_number,
    #         languages=args.languages,
    #         write_as_words=args.write_numbers_as_words,
    #         template=args.prompt_template,
    #     )
    #
    # ----------------------------------------------------------------------- #

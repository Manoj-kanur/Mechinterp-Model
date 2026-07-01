"""Prompt dataset for inference: builds the prompts that main.py runs the model on.

A "prompt" is just the input string you feed the model. The model reads it, generates a
continuation, and we capture its internal activations while it does so. So your prompts ARE
your experiment: they decide what behaviour you get to study. This file is where you define them.

You implement `PromptDataset.generate_prompts`. A small self-contained example (single-digit
addition) is shown in comments -- uncomment it to see the pipeline run, then replace it with
prompts for your own task.
"""
import random


class PromptDataset:
    """A collection of prompts to run activation-extraction inference on.

    `inference.run` iterates over `self.prompts`, a list of {"prompt": str, "metadata": dict}
    entries; that is the only interface the rest of the pipeline relies on. "prompt" is the string
    fed to the model; "metadata" is an optional free-form dict that can carry that prompt's ground truth
    (e.g. the operands and correct answer) so you can score the model's output downstream. It's
    passed straight through to each result row and never inspected by the pipeline, so put whatever
    you need in it (or leave it empty).
    """

    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self.prompts: list[dict] = []

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    @classmethod
    def generate_prompts(cls, num_prompts: int) -> "PromptDataset":
        """Build the list of prompts to run inference on.

        Args:
            num_prompts: How many prompts to generate (comes from `--num-prompts`).

        Returns:
            A PromptDataset whose `.prompts` is a list of `num_prompts` {"prompt": str,
            "metadata": dict} entries.
        """
        # Define digit maps for multilingual support (matching create_dataset.py)
        DIGIT_MAPS = {
            "english": list("0123456789"),
            "hindi": list("०१२३४५६७८९"),
            "mandarin": list("零一二三四५६७८९"),
        }
        LANGUAGES = list(DIGIT_MAPS.keys())
        PAD_WIDTH = 3  # zero-pad operands to 3 digits
        
        def render_number(n: int, lang: str, width: int = PAD_WIDTH) -> str:
            """Render an integer digit-by-digit in the target language's numeral script."""
            digits = DIGIT_MAPS[lang]
            padded = str(n).zfill(width)
            return "".join(digits[int(d)] for d in padded)
        
        def render_answer(n: int, lang: str) -> str:
            """Render the answer with REVERSED digit order and NO zero-padding."""
            digits = DIGIT_MAPS[lang]
            return "".join(digits[int(d)] for d in str(n)[::-1])
        
        def render_equation(a: int, b: int, lang: str) -> str:
            """Build the prompt ending with '=' so model generates the answer."""
            ra, rb = render_number(a, lang), render_number(b, lang)
            return f"{ra}+{rb}="
        
        instance = cls()
        for _ in range(num_prompts):
            # Randomly select language for this prompt
            lang = random.choice(LANGUAGES)
            
            # Generate two random operands (0-999 to match training data)
            a, b = random.randint(0, 999), random.randint(0, 999)
            answer = a + b
            
            # Build the prompt (ends with "=" so model generates the answer)
            prompt = render_equation(a, b, lang)
            
            # Store metadata for ground-truth comparison during scoring
            instance.prompts.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "a": a,
                        "b": b,
                        "answer": render_answer(answer, lang),
                        "language": lang,
                        "eng_question": f"{a}+{b}",
                        "eng_answer": str(answer),
                    },
                }
            )
        
        return instance
        raise NotImplementedError("Implement PromptDataset.generate_prompts() -- see the example in comments.")

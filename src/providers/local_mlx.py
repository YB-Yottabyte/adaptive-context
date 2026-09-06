"""Local MLX provider for Apple Silicon inference."""

from __future__ import annotations

from typing import Any, ClassVar

from providers.base import GenerationResult, LLMProvider, TextCallback

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
SYSTEM_PROMPT = (
    "You are a precise repository-level Python debugging assistant. "
    "Follow the user's requested output schema exactly and do not repeat the input."
)
RESPONSE_PREFIX = "Evidence considered:\n"


class LocalMLXProvider(LLMProvider):
    """Generate debugging responses locally with an MLX Hugging Face model."""

    _loaded_models: ClassVar[dict[str, tuple[Any, Any]]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL, max_tokens: int = 512) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.model_name = model_name
        self.max_tokens = max_tokens

    def _load_model(self) -> tuple[Any, Any]:
        """Load and cache each requested model once per Python process."""
        if self.model_name not in self._loaded_models:
            from huggingface_hub.utils import disable_progress_bars
            from mlx_lm import load

            with disable_progress_bars():
                model, tokenizer = load(self.model_name)
            if tokenizer.eos_token is not None:
                tokenizer.add_eos_token(tokenizer.eos_token)
            self._loaded_models[self.model_name] = model, tokenizer
        return self._loaded_models[self.model_name]

    def generate(
        self,
        prompt: str,
        *,
        on_text: TextCallback | None = None,
    ) -> GenerationResult:
        """Generate deterministically and report visible text as it arrives."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": RESPONSE_PREFIX},
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=True,
            return_dict=False,
        )

        response_parts: list[str] = []
        output_tokens = 0
        response_started = False
        for response in stream_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            response_parts.append(response.text)
            output_tokens = response.generation_tokens
            if response.text and on_text is not None:
                visible_text = (
                    f"{RESPONSE_PREFIX}{response.text}"
                    if not response_started
                    else response.text
                )
                on_text(visible_text)
            if response.text:
                response_started = True

        return GenerationResult(
            text=f"{RESPONSE_PREFIX}{''.join(response_parts).strip()}",
            input_tokens=len(prompt_tokens),
            output_tokens=output_tokens,
        )

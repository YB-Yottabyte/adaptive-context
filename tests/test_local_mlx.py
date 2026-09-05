from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from providers.local_mlx import RESPONSE_PREFIX, SYSTEM_PROMPT, LocalMLXProvider


def test_non_positive_max_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        LocalMLXProvider(max_tokens=0)


def test_blank_prompt_is_rejected_before_loading_mlx() -> None:
    provider = LocalMLXProvider()

    with (
        patch.object(provider, "_load_model") as load_model,
        pytest.raises(ValueError, match="prompt must not be empty"),
    ):
        provider.generate(" \n ")

    load_model.assert_not_called()


def test_local_generation_streams_visible_text() -> None:
    tokenizer = SimpleNamespace(
        apply_chat_template=Mock(return_value=[1, 2, 3]),
    )
    responses = iter(
        [
            SimpleNamespace(text="- payment.py calls the helper\n", generation_tokens=5),
            SimpleNamespace(text="Conclusion:\nReturn its value.", generation_tokens=9),
        ]
    )
    stream_generate = Mock(return_value=responses)
    make_sampler = Mock(return_value="sampler")
    mlx_lm = SimpleNamespace(stream_generate=stream_generate)
    sample_utils = SimpleNamespace(make_sampler=make_sampler)
    provider = LocalMLXProvider(model_name="local-test-model", max_tokens=32)
    received: list[str] = []

    with (
        patch.object(provider, "_load_model", return_value=("model", tokenizer)),
        patch.dict(
            "sys.modules",
            {
                "mlx_lm": mlx_lm,
                "mlx_lm.sample_utils": sample_utils,
            },
        ),
    ):
        result = provider.generate("debug this", on_text=received.append)

    assert received == [
        "Evidence considered:\n- payment.py calls the helper\n",
        "Conclusion:\nReturn its value.",
    ]
    assert result.text == (
        "Evidence considered:\n- payment.py calls the helper\n"
        "Conclusion:\nReturn its value."
    )
    assert result.input_tokens == 3
    assert result.output_tokens == 9
    tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "debug this"},
            {"role": "assistant", "content": RESPONSE_PREFIX},
        ],
        add_generation_prompt=False,
        continue_final_message=True,
        tokenize=True,
        return_dict=False,
    )
    make_sampler.assert_called_once_with(temp=0.0)
    stream_generate.assert_called_once_with(
        "model",
        tokenizer,
        [1, 2, 3],
        max_tokens=32,
        sampler="sampler",
    )


def test_model_loading_is_cached_by_model_name() -> None:
    tokenizer = SimpleNamespace(
        eos_token="</s>",
        add_eos_token=Mock(),
    )
    load = Mock(return_value=("model", tokenizer))
    mlx_lm = SimpleNamespace(load=load)
    first = LocalMLXProvider(model_name="cached-model")
    second = LocalMLXProvider(model_name="cached-model")

    LocalMLXProvider._loaded_models.pop("cached-model", None)
    try:
        with patch.dict("sys.modules", {"mlx_lm": mlx_lm}):
            assert first._load_model() == ("model", tokenizer)
            assert second._load_model() == ("model", tokenizer)
    finally:
        LocalMLXProvider._loaded_models.pop("cached-model", None)

    load.assert_called_once_with("cached-model")
    tokenizer.add_eos_token.assert_called_once_with("</s>")

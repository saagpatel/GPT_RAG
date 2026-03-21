from __future__ import annotations

import sys
from types import SimpleNamespace

from gpt_rag.config import load_settings
from gpt_rag.reranking import CrossEncoderReranker, Qwen3Reranker, build_reranker


def test_cross_encoder_reranker_stays_local_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, local_files_only: bool) -> None:
            captured["model_name"] = model_name
            captured["local_files_only"] = local_files_only

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    reranker = CrossEncoderReranker("cross-encoder/ms-marco")
    scores = reranker.score("socket timeout", ["chunk one"])

    assert scores == [0.5]
    assert captured["model_name"] == "cross-encoder/ms-marco"
    assert captured["local_files_only"] is True


def test_qwen3_reranker_uses_local_transformers_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = values

        def to(self, device):
            captured.setdefault("devices", []).append(device)
            return self

        def __getitem__(self, key):
            if isinstance(key, tuple):
                row_index, _last_token, token_index = key
                return self.values[row_index][token_index]
            return self.values[key]

    class FakeTokenizer:
        pad_token = None
        eos_token = "<|im_end|>"
        pad_token_id = 151645

        def __init__(self) -> None:
            self.padding_side = "left"

        def encode(self, text: str, add_special_tokens: bool = False):
            return [len(text) % 11 + 1]

        def __call__(self, value, **kwargs):
            if value == "yes":
                return SimpleNamespace(input_ids=[42])
            if value == "no":
                return SimpleNamespace(input_ids=[24])
            items = value if isinstance(value, list) else [value]
            captured["formatted_pairs"] = list(items)
            return {"input_ids": [[101, 102] for _ in items]}

        def pad(self, inputs, padding: bool, return_tensors: str):
            assert padding is True
            assert return_tensors == "pt"
            captured["padded_input_ids"] = [list(value) for value in inputs["input_ids"]]
            return {
                "input_ids": FakeTensor(inputs["input_ids"]),
            }

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.config = SimpleNamespace(pad_token_id=None)

        def eval(self):
            return self

        def __call__(self, **kwargs):
            captured["model_kwargs"] = kwargs
            return SimpleNamespace(
                logits=FakeTensor(
                    [
                        {24: 0.0, 42: 2.0},
                        {24: 1.5, 42: 0.0},
                    ]
                )
            )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, *, padding_side: str, local_files_only: bool):
            captured["tokenizer_model_name"] = model_name
            captured["tokenizer_local_files_only"] = local_files_only
            captured["tokenizer_padding_side"] = padding_side
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, *, local_files_only: bool):
            captured["model_model_name"] = model_name
            captured["model_local_files_only"] = local_files_only
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
        ),
    )

    reranker = Qwen3Reranker("Qwen/Qwen3-Reranker-4B")
    scores = reranker.score("socket timeout", ["socket timeout guide", "widget notes"])

    assert scores[0] > scores[1]
    assert captured["tokenizer_model_name"] == "Qwen/Qwen3-Reranker-4B"
    assert captured["tokenizer_local_files_only"] is True
    assert captured["model_model_name"] == "Qwen/Qwen3-Reranker-4B"
    assert captured["model_local_files_only"] is True
    assert captured["tokenizer_padding_side"] == "left"
    assert captured["formatted_pairs"][0].startswith(
        "<Instruct>: Given a local knowledge-base query"
    )
    assert reranker._model.config.pad_token_id == 151645


def test_build_reranker_uses_qwen_backend_for_default_model() -> None:
    reranker = build_reranker(load_settings())

    assert isinstance(reranker, Qwen3Reranker)

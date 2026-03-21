"""Local reranking interface."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gpt_rag.config import Settings


class RerankerError(RuntimeError):
    """Base error for local reranking backends."""


class RerankerDependencyError(RerankerError):
    """Raised when the local reranker dependency is unavailable."""


class RerankerModelNotAvailableError(RerankerError):
    """Raised when the configured reranker model is unavailable locally."""


@dataclass(slots=True)
class RerankerCacheReport:
    model_name: str
    cache_root: Path
    repo_path: Path
    snapshot_path: Path | None
    available: bool
    missing_files: list[str]
    incomplete_files: list[str]
    dependencies_available: bool = True
    dependency_error: str | None = None


class LocalReranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one reranker score per candidate text."""


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerDependencyError(
                "Local reranking requires the optional reranker dependencies. "
                "Install them with `python -m pip install -e \".[reranker]\"`."
            ) from exc

        try:
            self._model = CrossEncoder(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        except OSError as exc:
            raise RerankerModelNotAvailableError(
                f"Local reranker model {self.model_name!r} is not available locally. "
                "Download it into your local model cache first; no cloud fallback is used."
            ) from exc
        self._ensure_padding_configuration()
        return self._model

    def _ensure_padding_configuration(self) -> None:
        if self._model is None:
            return

        tokenizer = getattr(self._model, "tokenizer", None)
        model = getattr(self._model, "model", None)
        config = getattr(model, "config", None)
        if tokenizer is None or config is None:
            return

        pad_token = getattr(tokenizer, "pad_token", None)
        eos_token = getattr(tokenizer, "eos_token", None)
        if pad_token is None and eos_token is not None:
            tokenizer.pad_token = eos_token

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if getattr(config, "pad_token_id", None) is None and pad_token_id is not None:
            config.pad_token_id = pad_token_id

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        items = list(texts)
        if not items:
            return []
        model = self._load_model()
        pairs = [(query, text) for text in items]
        return [float(score) for score in model.predict(pairs)]


QWEN3_RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_RERANK_INSTRUCTION = (
    "Given a local knowledge-base query, retrieve relevant passages that answer the query."
)
QWEN3_RERANK_MAX_LENGTH = 8192


class Qwen3Reranker:
    def __init__(self, model_name: str, *, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._yes_token_id: int | None = None
        self._no_token_id: int | None = None

    def _load_model(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RerankerDependencyError(
                "Local reranking requires the optional reranker dependencies. "
                "Install them with `python -m pip install -e \".[reranker]\"`."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                local_files_only=self.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            ).eval()
        except OSError as exc:
            raise RerankerModelNotAvailableError(
                f"Local reranker model {self.model_name!r} is not available locally. "
                "Download it into your local model cache first; no cloud fallback is used."
            ) from exc

        if getattr(tokenizer, "pad_token", None) is None and getattr(
            tokenizer, "eos_token", None
        ) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if (
            getattr(model.config, "pad_token_id", None) is None
            and getattr(tokenizer, "pad_token_id", None) is not None
        ):
            model.config.pad_token_id = tokenizer.pad_token_id

        prefix = (
            "<|im_start|>system\n"
            f"{QWEN3_RERANK_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(
            QWEN3_RERANK_SUFFIX,
            add_special_tokens=False,
        )
        self._yes_token_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
        self._no_token_id = tokenizer("no", add_special_tokens=False).input_ids[0]
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        items = list(texts)
        if not items:
            return []

        model, tokenizer = self._load_model()
        prefix_tokens = self._prefix_tokens or []
        suffix_tokens = self._suffix_tokens or []
        max_body_tokens = max(1, QWEN3_RERANK_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens))

        formatted_pairs = [
            self._format_pair(query=query, document=text)
            for text in items
        ]
        inputs = tokenizer(
            formatted_pairs,
            add_special_tokens=False,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_body_tokens,
        )
        for index, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = prefix_tokens + token_ids + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        for key, value in inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(model.device)

        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[:, -1, :]
        yes_token_id = self._yes_token_id
        no_token_id = self._no_token_id
        if yes_token_id is None or no_token_id is None:
            raise RerankerError("Reranker yes/no token ids were not initialized.")

        scores: list[float] = []
        for index in range(len(items)):
            true_logit = _logit_value(logits, index, yes_token_id)
            false_logit = _logit_value(logits, index, no_token_id)
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            scores.append(true_score / (true_score + false_score))
        return scores

    @staticmethod
    def _format_pair(*, query: str, document: str) -> str:
        return (
            f"<Instruct>: {QWEN3_RERANK_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )


def _logit_value(logits: object, row_index: int, token_index: int) -> float:
    try:
        value = logits[row_index, token_index]
    except (TypeError, KeyError, IndexError):
        value = logits[row_index][token_index]
    if hasattr(value, "detach"):
        value = value.detach()
    return float(value)


def inspect_reranker_cache(model_name: str) -> RerankerCacheReport:
    cache_root = _huggingface_cache_root()
    repo_path = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_path / "snapshots"
    snapshot_paths = sorted(
        [path for path in snapshots_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    ) if snapshots_dir.exists() else []
    snapshot_path = snapshot_paths[-1] if snapshot_paths else None
    incomplete_files = (
        sorted(
            str(path.relative_to(repo_path))
            for path in repo_path.rglob("*.incomplete")
            if path.is_file()
        )
        if repo_path.exists()
        else []
    )
    missing_files = _missing_reranker_files(model_name, snapshot_path)
    available = snapshot_path is not None and not missing_files and not incomplete_files
    dependencies_available, dependency_error = _inspect_reranker_dependencies(model_name)
    return RerankerCacheReport(
        model_name=model_name,
        cache_root=cache_root,
        repo_path=repo_path,
        snapshot_path=snapshot_path,
        available=available,
        missing_files=missing_files,
        incomplete_files=incomplete_files,
        dependencies_available=dependencies_available,
        dependency_error=dependency_error,
    )


def _inspect_reranker_dependencies(model_name: str) -> tuple[bool, str | None]:
    try:
        if model_name.startswith("Qwen/Qwen3-Reranker-"):
            __import__("transformers")
        else:
            __import__("sentence_transformers")
    except ImportError:
        return (
            False,
            "Local reranking requires the optional reranker dependencies. "
            "Install them with `python -m pip install -e \".[reranker]\"`.",
        )
    return True, None


def _huggingface_cache_root() -> Path:
    explicit_hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit_hub_cache:
        return Path(explicit_hub_cache).expanduser()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _missing_reranker_files(model_name: str, snapshot_path: Path | None) -> list[str]:
    if snapshot_path is None:
        return ["snapshot"]

    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = [
        filename for filename in required_files if not (snapshot_path / filename).exists()
    ]

    if model_name.startswith("Qwen/Qwen3-Reranker-"):
        shard_files = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        missing_files.extend(
            filename for filename in shard_files if not (snapshot_path / filename).exists()
        )
        return missing_files

    if not any(
        (snapshot_path / filename).exists()
        for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    ):
        missing_files.append("model weights")
    return missing_files


def build_reranker(settings: Settings) -> LocalReranker:
    if settings.reranker_model.startswith("Qwen/Qwen3-Reranker-"):
        return Qwen3Reranker(settings.reranker_model)
    return CrossEncoderReranker(settings.reranker_model)

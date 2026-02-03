from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

@dataclass
class ApiConfig:
    default_method: str
    default_k: int
    device: str
    max_k: int

@dataclass
class ModelCfg:
    hf_name: str
    local_path: Optional[str]
    type: str  # finetune | hf

@dataclass
class FinetuneCfg:
    max_length: int
    max_candidates_per_doc: int
    ngram_max: int
    min_token_len: int
    score_threshold: float

@dataclass
class AppConfig:
    api: ApiConfig
    models: Dict[str, ModelCfg]
    finetune: FinetuneCfg

def load_yaml_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe config YAML: {path}")

    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    api_raw = raw.get("api", {})
    api = ApiConfig(
        default_method=api_raw.get("default_method", "finetune"),
        default_k=int(api_raw.get("default_k", 15)),
        device=str(api_raw.get("device", "auto")),
        max_k=int(api_raw.get("max_k", 50)),
    )

    models_raw = raw.get("models", {})
    models: Dict[str, ModelCfg] = {}
    for name, m in models_raw.items():
        models[name] = ModelCfg(
            hf_name=str(m.get("hf_name", "")),
            local_path=m.get("local_path"),
            type=str(m.get("type", "hf")),
        )

    ft_raw = raw.get("finetune", {})
    finetune = FinetuneCfg(
        max_length=int(ft_raw.get("max_length", 256)),
        max_candidates_per_doc=int(ft_raw.get("max_candidates_per_doc", 200)),
        ngram_max=int(ft_raw.get("ngram_max", 3)),
        min_token_len=int(ft_raw.get("min_token_len", 3)),
        score_threshold=float(ft_raw.get("score_threshold", 0.05)),
    )

    return AppConfig(api=api, models=models, finetune=finetune)

def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device in {"cpu", "cuda", "mps"}:
        return device

    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

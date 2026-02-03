from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .config import AppConfig, resolve_device

@dataclass
class LoadedModel:
    name: str
    tokenizer: object
    model: object
    device: str

class ModelRegistry:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.api.device)
        self._cache: Dict[str, LoadedModel] = {}

    def get(self, model_key: str) -> LoadedModel:
        if model_key in self._cache:
            return self._cache[model_key]

        if model_key not in self.cfg.models:
            raise ValueError(f"Modelo '{model_key}' no existe en config.yml (models: ...)")

        mcfg = self.cfg.models[model_key]
        tokenizer, model, src = self._load_tokenizer_and_model(mcfg.hf_name, mcfg.local_path)
        print(f"Loading model {model_key} from {src}")
        model.to(self.device)
        model.eval()

        loaded = LoadedModel(name=model_key, tokenizer=tokenizer, model=model, device=self.device)
        self._cache[model_key] = loaded
        return loaded

    def _load_tokenizer_and_model(self, hf_name: str, local_path: Optional[str]):
        """
        Regla:
          - si local_path existe y tiene pesos -> carga desde local
          - si local_path apunta a checkpoint inválido -> busca otro checkpoint válido
          - si no -> carga desde HF (si hf_name existe)
        """
        src_path: Optional[Path] = None
        if local_path:
            p = Path(local_path)
            if p.exists() and p.is_dir():
                if self._has_weights(p):
                    src_path = p
                else:
                    if p.name.startswith("checkpoint-"):
                        search_root = p.parent
                    else:
                        search_root = p
                    best = self._find_best_checkpoint(search_root)
                    if best is not None:
                        src_path = best

        src = str(src_path) if src_path is not None else hf_name
        if not src:
            raise ValueError("hf_name vacío y local_path no válido. Revisa config.yml")

        tokenizer = AutoTokenizer.from_pretrained(src)
        # Clasificación binaria para finetune (2 labels). Si es HF base sin head binaria,
        # esto puede fallar: en ese caso usa tus checkpoints finetuneados o un modelo con head.
        model = AutoModelForSequenceClassification.from_pretrained(src)
        return tokenizer, model, src

    @staticmethod
    def _has_weights(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        for name in ("model.safetensors", "pytorch_model.bin"):
            p = path / name
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return True
        return False

    @staticmethod
    def _find_best_checkpoint(root: Path) -> Optional[Path]:
        if not root.exists() or not root.is_dir():
            return None
        best_step = -1
        best_path: Optional[Path] = None
        for p in root.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            if not ModelRegistry._has_weights(p):
                continue
            try:
                step = int(p.name.split("-")[-1])
            except ValueError:
                step = -1
            if step > best_step:
                best_step = step
                best_path = p
        return best_path

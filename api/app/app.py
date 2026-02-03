from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional, Dict, Any
import os

from .schemas import DocIn, PredictResponse, DocOut, MetaOut
from .config import load_yaml_config
from .model_registry import ModelRegistry
from .keyword_extraction import finetune_extract_keywords, baseline_extract_keywords

def create_app() -> FastAPI:
    app = FastAPI(title="TFG Keyword Extraction API", version="0.1")

    # Config path: ENV > default
    cfg_path = os.environ.get("TFG_API_CONFIG", "config/api_config.yml")
    cfg = load_yaml_config(cfg_path)
    registry = ModelRegistry(cfg)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        return {"name": "TFG Keyword Extraction API", "status": "ok"}

    @app.post("/{method}/{k}", response_model=PredictResponse)
    def predict(
        method: str,
        k: int,
        docs: List[DocIn],
        model: str = Query(default=None, description="Clave de modelo definida en config.yml (models: ...)"),
    ):
        method = (method or cfg.api.default_method).lower()
        if k <= 0:
            raise HTTPException(status_code=400, detail="K debe ser > 0")
        if k > cfg.api.max_k:
            raise HTTPException(status_code=400, detail=f"K demasiado grande (max_k={cfg.api.max_k})")

        if method not in {"finetune", "baseline"}:
            raise HTTPException(status_code=400, detail=f"Método '{method}' no soportado (por ahora: finetune, baseline)")

        results: List[DocOut] = []

        # --------------------
        # BASELINE (TF-IDF)
        # --------------------
        if method == "baseline":
            for d in docs:
                text = d.text.strip()
                if not text:
                    results.append(DocOut(keywords=[]))
                    continue
                try:
                    kws = baseline_extract_keywords(cfg, text, k)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error en baseline TF-IDF: {e}")
                results.append(DocOut(keywords=kws))

            meta = MetaOut(method=method, k=k, model="baseline-tfidf", device="cpu")
            return PredictResponse(results=results, meta=meta)

        # --------------------
        # FINETUNE
        # --------------------
        model_key = model or ("roberta" if "roberta" in cfg.models else next(iter(cfg.models.keys()), None))
        if not model_key:
            raise HTTPException(status_code=500, detail="No hay modelos definidos en config.yml")

        try:
            mdl = registry.get(model_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo '{model_key}': {e}")

        for d in docs:
            text = d.text.strip()
            if not text:
                results.append(DocOut(keywords=[]))
                continue
            try:
                kws = finetune_extract_keywords(mdl, cfg, text, k)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error procesando documento: {e}")
            results.append(DocOut(keywords=kws))

        meta = MetaOut(method=method, k=k, model=model_key, device=mdl.device)
        return PredictResponse(results=results, meta=meta)

    return app

app = create_app()

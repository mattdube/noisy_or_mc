"""
phase4_api.py — FastAPI Deployment Layer
=========================================
Production-ready REST API wrapping the scorer.

Design principles:
  - Param store loaded ONCE at startup (not per request)
  - /reload endpoint for hot-swapping updated params without restart
  - Input validation via Pydantic
  - Async-compatible (non-blocking for IO)
  - JSON-serializable responses only

Dependencies:
    pip install fastapi uvicorn pydantic

Run locally:
    uvicorn phase4_api:app --host 0.0.0.0 --port 8000 --reload

Docker:
    FROM python:3.11-slim
    COPY . /app
    WORKDIR /app
    RUN pip install fastapi uvicorn numpy pandas scipy
    CMD ["uvicorn", "phase4_api:app", "--host", "0.0.0.0", "--port", "8000"]

Example API call:
    curl -X POST http://localhost:8000/score \
         -H "Content-Type: application/json" \
         -d '{"matches": [1,0,0,1,0,0,1,0]}'
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any

# --- Guard: FastAPI is optional (don't break import if not installed) ---
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, field_validator, model_validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[phase4_api] FastAPI not installed. Install: pip install fastapi uvicorn")
    print("             Standalone scorer functions are still usable.\n")

from phase3_scorer import score_single, reload_store, load_store, DEFAULT_N_MC


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:

    class ScoreRequest(BaseModel):
        matches: list[int]
        n_mc:    int = DEFAULT_N_MC
        seed:    int = 0

        @field_validator("matches")
        @classmethod
        def validate_matches(cls, v):
            if not all(x in (0, 1) for x in v):
                raise ValueError("matches must contain only 0 or 1")
            return v

        @model_validator(mode="after")
        def check_length(self):
            store = load_store()
            expected = len(store["source_order"])
            if len(self.matches) != expected:
                raise ValueError(
                    f"matches must have {expected} elements (one per source), "
                    f"got {len(self.matches)}"
                )
            return self


    class BatchScoreRequest(BaseModel):
        records: list[list[int]]  # [[0,1,0,...], [1,0,1,...], ...]
        n_mc:    int = 500        # lower default for batch (speed)
        seed:    int = 0

        @model_validator(mode="after")
        def check_shape(self):
            store = load_store()
            expected = len(store["source_order"])
            for i, row in enumerate(self.records):
                if len(row) != expected:
                    raise ValueError(f"Record {i} has {len(row)} values, expected {expected}")
                if not all(x in (0, 1) for x in row):
                    raise ValueError(f"Record {i} contains values outside {{0, 1}}")
            return self


    class ScoreResponse(BaseModel):
        risk_point:     float
        risk_p50:       float
        risk_p10:       float
        risk_p90:       float
        risk_std:       float
        impact_width:   float
        primary_driver: str
        matched_sources: str
        impacts:        dict[str, float]
        latency_ms:     float


    class BatchScoreResponse(BaseModel):
        n_records:  int
        results:    list[dict[str, Any]]
        latency_ms: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:

    app = FastAPI(
        title="Noisy-OR Risk Scorer API",
        description=(
            "Probabilistic risk scoring using Noisy-OR model with Beta priors. "
            "Returns point estimates, MC-based credible intervals, and driver attribution."
        ),
        version="1.0.0",
    )

    # Load param store at startup (cached in phase3_scorer module)
    @app.on_event("startup")
    def startup():
        store = load_store()
        print(f"[API] Param store loaded. Version: {store['version']}")
        print(f"[API] Sources: {store['source_order']}")


    @app.get("/health")
    def health():
        """Liveness check."""
        store = load_store()
        return {
            "status":          "ok",
            "param_version":   store["version"],
            "source_order":    store["source_order"],
        }


    @app.post("/reload")
    def reload_params(path: str = "params/param_store.json"):
        """
        Hot-reload updated parameters without restarting.
        Call this after running phase1_priors.py to update source confidence.
        """
        store = reload_store(path)
        return {"status": "reloaded", "version": store["version"]}


    @app.post("/score", response_model=ScoreResponse)
    def score_endpoint(req: ScoreRequest):
        """Score a single record."""
        t0 = time.perf_counter()

        match_vector = np.array(req.matches, dtype=np.float64)
        result = score_single(match_vector, n_mc=req.n_mc, seed=req.seed)

        latency = (time.perf_counter() - t0) * 1000
        return ScoreResponse(
            risk_point     = result["risk_point"],
            risk_p50       = result["risk_p50"],
            risk_p10       = result["risk_p10"],
            risk_p90       = result["risk_p90"],
            risk_std       = result["risk_std"],
            impact_width   = result["impact_width"],
            primary_driver = result["primary_driver"],
            matched_sources= result["matched_sources"],
            impacts        = result["impacts"],
            latency_ms     = round(latency, 2),
        )


    @app.post("/score/batch", response_model=BatchScoreResponse)
    def score_batch_endpoint(req: BatchScoreRequest):
        """
        Score multiple records in one call.
        More efficient than calling /score N times.
        """
        from phase3_scorer import score as score_batch

        t0 = time.perf_counter()
        match_matrix = np.array(req.records, dtype=np.float64)
        df = score_batch(match_matrix, n_mc=req.n_mc, seed=req.seed)
        results = df.to_dict(orient="records")
        latency = (time.perf_counter() - t0) * 1000

        return BatchScoreResponse(
            n_records  = len(results),
            results    = results,
            latency_ms = round(latency, 2),
        )


    @app.get("/params/summary")
    def params_summary():
        """Return current parameter store metadata and source stats."""
        store = load_store()
        return {
            "version":      store["version"],
            "created_utc":  store.get("created_utc"),
            "sources": {
                s: {
                    "mean":         store["sources"][s]["mean"],
                    "std":          store["sources"][s]["std"],
                    "ci90":         [store["sources"][s]["ci90_lo"],
                                     store["sources"][s]["ci90_hi"]],
                    "concentration": store["sources"][s]["expert_concentration"],
                }
                for s in store["source_order"]
            },
            "leak": {
                "mean": store["leak"]["mean"],
                "std":  store["leak"]["std"],
            }
        }


# ---------------------------------------------------------------------------
# Standalone demo (no FastAPI required)
# ---------------------------------------------------------------------------
def demo_without_api():
    """
    Demonstrates the scorer in API-equivalent mode without FastAPI.
    Use this to test scoring logic in environments where FastAPI isn't installed.
    """
    from phase1_priors import load_param_store
    store = load_param_store("params/param_store.json")

    print("=== API Demo (no FastAPI) ===\n")

    # Single record
    test_cases = [
        ([1, 0, 0, 1, 0, 0, 1, 0], "S1+S4+S7 matched (high risk mix)"),
        ([0, 0, 0, 0, 0, 0, 0, 0], "No matches (baseline)"),
        ([1, 1, 1, 1, 1, 1, 1, 1], "All matched (max risk)"),
        ([0, 1, 0, 0, 0, 0, 0, 0], "Only S2 matched (low confidence source)"),
        ([0, 0, 0, 0, 0, 0, 1, 0], "Only S7 matched (high-risk, high-confidence)"),
    ]

    for matches, desc in test_cases:
        t0 = time.perf_counter()
        result = score_single(np.array(matches, dtype=float), store=store, n_mc=2000)
        ms = (time.perf_counter() - t0) * 1000

        print(f"  {desc}")
        print(f"    risk_point  = {result['risk_point']:.4f}")
        print(f"    CI 80%      = [{result['risk_p10']:.4f}, {result['risk_p90']:.4f}]  "
              f"width={result['impact_width']:.4f}")
        print(f"    primary driver: {result['primary_driver']}")
        print(f"    latency: {ms:.1f}ms\n")


if __name__ == "__main__":
    import os
    if os.path.exists("params/param_store.json"):
        demo_without_api()
    else:
        print("Run phase1_priors.py first to create the parameter store.")
    
    if FASTAPI_AVAILABLE:
        print("\nTo start the API server:")
        print("  uvicorn phase4_api:app --host 0.0.0.0 --port 8000")

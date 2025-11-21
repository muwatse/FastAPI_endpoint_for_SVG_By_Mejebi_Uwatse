# app/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict

from .models import CropSubmitPayload
from .job_manager import create_job, JOBS
from .metrics import router as metrics_router, CROP_SUBMIT_REQUESTS, CACHE_HITS, CACHE_MISSES
from .db_cache import make_cache_key, get_cached_result
from app.logger import console

app = FastAPI(title="Frontal Crop API", version="1.0.0")

# Include /metrics endpoint
app.include_router(metrics_router)


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "ok", "message": "Face SVG API"}


@app.post("/api/v1/frontal/crop/submit")
async def submit_crop(
    payload: CropSubmitPayload,
    fast: bool = Query(
        False,
        description="If true skip simulated 20s delay (for local testing)",
    ),
):
    """
    Submit payload and immediately return job id + pending.

    Body:
      {
        "image": "<base64>",
        "segmentation_map": "<base64>",
        "landmarks": [{ "x": ..., "y": ... }, ...],
        "original_width": 1024,  # optional
        "original_height": 1024  # optional
      }
    """
    payload_dict = payload.dict()

    if not payload_dict.get("segmentation_map"):
        raise HTTPException(status_code=422, detail="segmentation_map required")
    if not payload_dict.get("image"):
        raise HTTPException(status_code=422, detail="image required")

    # CHECK CACHE FIRST
    cache_key = make_cache_key(payload_dict)
    cached = get_cached_result(cache_key)
    
    if cached:
        console.log(f"[green]Cache HIT for key {cache_key[:12]}...[/green]")
        CACHE_HITS.inc()
        return JSONResponse(status_code=200, content=cached)
    
    # Cache MISS - proceed with job creation
    console.log(f"[yellow]Cache MISS for key {cache_key[:12]}...[/yellow]")
    CACHE_MISSES.inc()

    # Record that a request came in
    CROP_SUBMIT_REQUESTS.labels(fast=str(fast).lower()).inc()

    job_id = create_job(payload_dict, simulate_delay=not fast)
    console.log(f"[blue]Received job {job_id} (fast={fast})[/blue]")

    # Explicitly return 202 so the test script is happy
    return JSONResponse(status_code=202, content={"id": job_id, "status": "pending"})


@app.get("/api/v1/frontal/crop/status/{job_id}")
async def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    status = job.get("status")
    if status == "pending":
        return {"id": job_id, "status": "pending"}
    if status == "error":
        return {"id": job_id, "status": "error", "error": job.get("error")}

    # status == "done"
    return job["result"]

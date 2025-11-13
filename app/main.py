from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from .models import CropSubmitPayload
from .tasks import create_job, JOBS
from typing import Dict
from rich import print

app =FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Face SVG API placeholder"}

@app.post("/api/v1/frontal/crop/submit")
async def submit_crop(payload: CropSubmitPayload, fast: bool = Query(False, description="If true skip simulated 20s delay")):
    """
    Submit payload and immediately return job id + pending.
    Use query param fast=true to skip the 20s simulated wait for test/load.
    """
    payload_dict = payload.dict()
    if not payload_dict.get("segmentation_map"):
        raise HTTPException(status_code=422, detail="segmentation_map required")

    job_id = create_job(payload_dict, simulate_delay=not fast)
    print(f"[blue]Received job {job_id} (fast={fast})[/blue]")
    return JSONResponse(status_code=202, content={"id": job_id, "status": "pending"})

@app.get("/api/v1/frontal/crop/status/{job_id}")
async def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "pending":
        return {"id": job_id, "status": "pending"}
    if job["status"] == "error":
        return {"id": job_id, "status": "error", "error": job.get("error")}
    # done
    return job["result"]
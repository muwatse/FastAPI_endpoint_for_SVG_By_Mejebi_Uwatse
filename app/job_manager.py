# app/job_manager.py
"""
Job queue management for async face processing tasks.
Handles job creation, scheduling, and status tracking.
"""

import asyncio
import uuid
from typing import Dict, Any
from app.logger import console

from .processing import process_face_image
from .metrics import JOB_PROCESSING_SECONDS, JOBS_COMPLETED, JOBS_IN_FLIGHT

# In-memory job store
JOBS: Dict[str, Dict[str, Any]] = {}


async def process_job(job_id: str, payload: Dict[str, Any], simulate_delay: bool = True):
    """
    Background worker that delegates to the monolithic processor.

    Args:
        job_id: Unique job identifier
        payload: Dictionary containing image, segmentation_map, landmarks, etc.
        simulate_delay: If True, waits 20 seconds before processing
    """
    console.log(f"[yellow]Starting background processing for job {job_id}[/yellow]")

    try:
        if simulate_delay:
            await asyncio.sleep(20)

        # Time the job processing
        with JOB_PROCESSING_SECONDS.time():
            result = process_face_image(payload)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result

        # Metrics: job completed successfully
        JOBS_COMPLETED.labels(status="done").inc()
        JOBS_IN_FLIGHT.dec()

        console.log(f"[green]Job {job_id} done.[/green]")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

        # Metrics: job completed with error
        JOBS_COMPLETED.labels(status="error").inc()
        JOBS_IN_FLIGHT.dec()

        console.log(f"[red]Job {job_id} failed: {e}[/red]")


def create_job(payload: Dict[str, Any], simulate_delay: bool = True) -> str:
    """
    Create job entry and schedule async background task.

    Args:
        payload: Request payload containing image data
        simulate_delay: Whether to simulate 20s processing delay

    Returns:
        job_id: Unique identifier for tracking job status
    """
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "pending", "payload": payload}

    # Metrics: increase in flight jobs
    JOBS_IN_FLIGHT.inc()

    loop = asyncio.get_event_loop()
    loop.create_task(process_job(job_id, payload, simulate_delay=simulate_delay))

    return job_id

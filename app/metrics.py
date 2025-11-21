# app/metrics.py
"""
Prometheus metrics and /metrics endpoint for the FastAPI app.
"""

from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

# Count how many times /crop/submit is called, with label for fast mode
CROP_SUBMIT_REQUESTS = Counter(
    "frontal_crop_submit_requests_total",
    "Total number of /api/v1/frontal/crop/submit requests",
    ["fast"],
)

# Track how many jobs are currently in flight (pending or processing)
JOBS_IN_FLIGHT = Gauge(
    "frontal_crop_jobs_in_flight",
    "Number of frontal crop jobs currently not finished",
)

# Measure processing time per job
JOB_PROCESSING_SECONDS = Histogram(
    "frontal_crop_job_processing_seconds",
    "Time spent processing frontal crop jobs in seconds",
)

# Count jobs by final status
JOBS_COMPLETED = Counter(
    "frontal_crop_jobs_completed_total",
    "Total number of completed frontal crop jobs by status",
    ["status"],  # done, error
)
# Cache hit/miss tracking
CACHE_HITS = Counter(
    "frontal_crop_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES = Counter(
    "frontal_crop_cache_misses_total",
    "Total number of cache misses",
)


@router.get("/metrics")
def metrics() -> Response:
    """
    Expose Prometheus metrics in text format.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

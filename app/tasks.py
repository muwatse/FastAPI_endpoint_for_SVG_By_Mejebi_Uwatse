# app/tasks.py
import asyncio
import uuid
from typing import Dict, Any
from rich import print
from .utils import (
    base64_to_pil,
    pil_to_numpy_rgba,
    extract_contours_from_segmap,
    contours_to_svg,
    svg_to_base64,
)

# In-memory job store
JOBS: Dict[str, Dict[str, Any]] = {}

async def process_job(job_id: str, payload: Dict[str, Any], simulate_delay: bool = True):
    """Background worker: decode segmap -> extract contours -> build SVG overlay -> store result."""
    print(f"[yellow]Starting background processing for job {job_id}[/yellow]")
    try:
        if simulate_delay:
            await asyncio.sleep(20)

        seg_b64 = payload["segmentation_map"]
        pil_img = base64_to_pil(seg_b64)
        np_img = pil_to_numpy_rgba(pil_img)
        h, w = np_img.shape[:2]

        contours_by_color = extract_contours_from_segmap(np_img)

        # Embed original photo beneath the translucent overlay
        orig_b64 = payload.get("image")
        svg_str = contours_to_svg(contours_by_color, width=w, height=h, orig_image_b64=orig_b64)
        svg_b64 = svg_to_base64(svg_str, data_uri=True)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "svg": svg_b64,
            "mask_contours": contours_by_color,
        }
        print(f"[green]Job {job_id} done. Found {len(contours_by_color)} regions.[/green]")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        print(f"[red]Job {job_id} failed: {e}[/red]")

def create_job(payload: Dict[str, Any], simulate_delay: bool = True) -> str:
    """Create job entry and schedule async background task."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "pending", "payload": payload}
    loop = asyncio.get_event_loop()
    loop.create_task(process_job(job_id, payload, simulate_delay=simulate_delay))
    return job_id

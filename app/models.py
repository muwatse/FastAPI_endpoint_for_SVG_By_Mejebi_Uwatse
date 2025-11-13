# app/models.py
from pydantic import BaseModel
from typing import List, Dict, Any


class Point(BaseModel):
    x: float
    y: float


class CropSubmitPayload(BaseModel):
    image: str  # base64 image (original) - kept for completeness
    segmentation_map: str  # base64 encoded segmentation image
    landmarks: List[Point] = []


class JobStatus(BaseModel):
    id: str
    status: str


class JobResult(BaseModel):
    svg: str
    mask_contours: Dict[str, List[List[List[float]]]]  

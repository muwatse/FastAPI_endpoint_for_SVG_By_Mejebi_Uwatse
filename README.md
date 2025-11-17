Frontal Face SVG Processing Service

This project is a FastAPI-based microservice that accepts an image, segmentation map, and facial landmarks, processes the face in a background job, and returns SVG overlays of facial regions (forehead, nose, eyebrows, jawline, etc.).
It runs asynchronously, exposes Prometheus metrics, and uses Rich for clean console logs.

How to Install & Run

Run With Docker (Recommended)
1. Install Docker & Docker Compose

Download Docker Desktop (Windows/Mac) or use your package manager on Linux.

2. Build and start

From the project root:

docker-compose up --build


This starts:

Service	Port
FastAPI app	http://localhost:8000

Prometheus	http://localhost:9090
3. Test the API
Submit a job:
POST http://localhost:8000/api/v1/frontal/crop/submit


Body (example):

{
  "image": "<base64>",
  "segmentation_map": "<base64>",
  "landmarks": [...]
}


Response:

{ "id": "xxxx", "status": "pending" }

Check job status:
GET http://localhost:8000/api/v1/frontal/crop/status/{id}


Returns SVG + mask contours when done.

4. View metrics
http://localhost:8000/metrics


Prometheus dashboard:

http://localhost:9090


# app/db_cache.py
"""
Postgres backed cache for mask results.

Responsibilities:
- Manage a single Postgres connection
- Create mask_cache table if needed
- Provide make_cache_key, get_cached_result, store_cached_result
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from app.logger import console

_DB_CONN = None
_DB_AVAILABLE = False


def _get_connection():
    """
    Get or create a global Postgres connection.

    If Postgres is not reachable, this will disable the cache gracefully.
    """
    global _DB_CONN, _DB_AVAILABLE

    if _DB_AVAILABLE and _DB_CONN is not None and not _DB_CONN.closed:
        return _DB_CONN

    host = os.getenv("DB_HOST", "postgres")
    port = int(os.getenv("DB_PORT", "5432"))
    name = os.getenv("DB_NAME", "face_cache")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")

    try:
        _DB_CONN = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
            user=user,
            password=password,
        )
        _DB_CONN.autocommit = True
        _DB_AVAILABLE = True

        _init_cache_table(_DB_CONN)
        console.log("[green]Postgres cache connected[/green]")
        return _DB_CONN

    except Exception as exc:
        # On first failure, log; keep cache disabled
        if not _DB_AVAILABLE:
            console.log(
                f"[yellow]Postgres cache disabled (cannot connect: {exc})[/yellow]"
            )
        _DB_AVAILABLE = False
        _DB_CONN = None
        return None


def _init_cache_table(conn) -> None:
    """
    Create the mask_cache table if it does not exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mask_cache (
                id SERIAL PRIMARY KEY,
                cache_key TEXT UNIQUE NOT NULL,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )


def make_cache_key(payload: Dict[str, Any]) -> str:
    """
    Build a deterministic cache key from all inputs that affect the result.

    Currently:
      - image (base64)
      - segmentation_map (base64)
      - landmarks (list of {x, y})
      - original_width / original_height

    If you add options later (fast mode, variants, etc), include them here.
    """
    h = hashlib.sha256()

    image_b64 = payload.get("image") or ""
    seg_b64 = payload.get("segmentation_map") or ""
    landmarks = payload.get("landmarks") or []
    orig_w = payload.get("original_width")
    orig_h = payload.get("original_height")

    h.update(image_b64.encode("utf-8"))
    h.update(seg_b64.encode("utf-8"))
    h.update(json.dumps(landmarks, sort_keys=True).encode("utf-8"))
    h.update(json.dumps({"orig_w": orig_w, "orig_h": orig_h}).encode("utf-8"))

    return h.hexdigest()


def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Look up a cached result by cache_key.

    Returns:
        - dict with the result if found
        - None if not found or cache is unavailable
    """
    conn = _get_connection()
    if conn is None:
        return None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT result_json FROM mask_cache WHERE cache_key = %s",
                (cache_key,),
            )
            row = cur.fetchone()
            if not row:
                return None

            try:
                return json.loads(row["result_json"])
            except Exception as exc:
                console.log(
                    f"[yellow]Failed to decode cached JSON for key {cache_key}: {exc}[/yellow]"
                )
                return None
    except Exception as exc:
        console.log(f"[yellow]Cache lookup failed for key {cache_key}: {exc}[/yellow]")
        return None


def store_cached_result(cache_key: str, result: Dict[str, Any]) -> None:
    """
    Store or update a cached result for cache_key.

    If Postgres is unavailable this becomes a no-op.
    """
    conn = _get_connection()
    if conn is None:
        return

    payload_str = json.dumps(result)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mask_cache (cache_key, result_json)
                VALUES (%s, %s)
                ON CONFLICT (cache_key) DO UPDATE
                SET result_json = EXCLUDED.result_json
                """,
                (cache_key, payload_str),
            )
    except Exception as exc:
        console.log(f"[yellow]Failed to store cache entry {cache_key}: {exc}[/yellow]")

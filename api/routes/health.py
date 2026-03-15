"""
api/routes/health.py

Simple health check endpoint. No agent or DB dependencies — always responds.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "version": "0.1.0"}

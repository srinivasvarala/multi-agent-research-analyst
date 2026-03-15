"""
api/main.py

FastAPI application entry point.
Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.research import router as research_router
from api.routes.stream import router as stream_router
from core.config import get_settings

settings = get_settings()

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", host=settings.api_host, port=settings.api_port)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Multi-Agent Research Analyst",
    description="Answers financial research questions using SEC filings, earnings calls, and news.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(research_router)
app.include_router(stream_router)

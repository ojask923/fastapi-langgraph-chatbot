"""FastAPI Main Entry Point."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.routes import router as api_router


import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    try:
        from alembic.config import Config
        from alembic import command
        
        logger.info("Running database migrations...")
        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")
        logger.info("Database migrations complete.")
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}")
        
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="A clean, simple local chatbot running with FastAPI, LangGraph, and SQLite/PostgreSQL database persistence",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

"""FastAPI application factory for Mozart dashboard.

Provides the web server for job monitoring and control.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mozart.state.base import StateBackend
from mozart.state.json_backend import JsonStateBackend

# Module-level references for dependency injection
_state_backend: StateBackend | None = None
_templates: Jinja2Templates | None = None


def get_state_backend() -> StateBackend:
    """Get the configured state backend.

    Raises:
        RuntimeError: If backend not configured (app not started properly)
    """
    if _state_backend is None:
        raise RuntimeError("State backend not configured. Use create_app() with a backend.")
    return _state_backend


def get_templates() -> Jinja2Templates:
    """Get the configured Jinja2Templates instance.

    Raises:
        RuntimeError: If templates not configured (app not started properly)
    """
    if _templates is None:
        raise RuntimeError("Templates not configured. Use create_app() to initialize.")
    return _templates


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Manages startup/shutdown for async resources.
    """
    # Startup: nothing async to initialize for JSON backend
    # (SQLite backend could init connection pool here)
    yield
    # Shutdown: cleanup if needed
    pass


def create_app(
    state_backend: StateBackend | None = None,
    state_dir: Path | str | None = None,
    title: str = "Mozart Dashboard",
    version: str = "0.1.0",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        state_backend: Pre-configured state backend (takes precedence)
        state_dir: Directory for JSON state files (used if no backend provided)
        title: API title for OpenAPI docs
        version: API version
        cors_origins: Allowed CORS origins (defaults to all for development)

    Returns:
        Configured FastAPI application
    """
    global _state_backend

    # Configure state backend
    if state_backend is not None:
        _state_backend = state_backend
    elif state_dir is not None:
        _state_backend = JsonStateBackend(Path(state_dir))
    else:
        # Default to current directory/.mozart-state for development
        _state_backend = JsonStateBackend(Path.cwd() / ".mozart-state")

    # Create app
    app = FastAPI(
        title=title,
        version=version,
        description="REST API for Mozart job orchestration",
        lifespan=lifespan,
    )

    # CORS middleware
    # Default to localhost origins for security; use MOZART_DEV=1 for permissive mode
    import os
    if cors_origins:
        allowed_origins = cors_origins
    elif os.environ.get("MOZART_DEV") == "1":
        # Development mode: allow all origins
        allowed_origins = ["*"]
    else:
        # Production default: restrict to localhost
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure template and static file paths
    dashboard_dir = Path(__file__).parent
    templates_dir = dashboard_dir / "templates"
    static_dir = dashboard_dir / "static"

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Configure Jinja2 templates
    global _templates
    _templates = Jinja2Templates(directory=str(templates_dir))

    # Register routes
    from mozart.dashboard.routes import router as base_router
    from mozart.dashboard.routes.artifacts import router as artifacts_router
    from mozart.dashboard.routes.jobs import router as jobs_router
    from mozart.dashboard.routes.pages import router as pages_router
    from mozart.dashboard.routes.scores import router as scores_router
    from mozart.dashboard.routes.stream import router as stream_router

    app.include_router(base_router)
    app.include_router(jobs_router)
    app.include_router(artifacts_router)
    app.include_router(pages_router)
    app.include_router(scores_router)
    app.include_router(stream_router)

    # Health check endpoint (at root level)
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, Any]:
        """Health check endpoint.

        Returns basic service health status.
        """
        return {
            "status": "healthy",
            "version": version,
            "service": "mozart-dashboard",
        }

    return app


# Type alias for dependency injection
StateBackendDep = StateBackend

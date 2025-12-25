"""FastAPI application factory for Mozart dashboard.

Provides the web server for job monitoring and control.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mozart.state.base import StateBackend
from mozart.state.json_backend import JsonStateBackend

# Module-level state backend reference for dependency injection
_state_backend: StateBackend | None = None


def get_state_backend() -> StateBackend:
    """Get the configured state backend.

    Raises:
        RuntimeError: If backend not configured (app not started properly)
    """
    if _state_backend is None:
        raise RuntimeError("State backend not configured. Use create_app() with a backend.")
    return _state_backend


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
    allowed_origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from mozart.dashboard.routes import router
    app.include_router(router)

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

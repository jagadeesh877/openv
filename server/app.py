"""
server/app.py
=============
Entry point for multi-mode deployment.
This module exposes the FastAPI app and a main() launcher
so it can be discovered via [project.scripts] in pyproject.toml.
"""
import os
import sys

# Ensure the project root is on the path so all imports resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Re-export the FastAPI app created in main.py
from main import app  # noqa: F401  (re-export)


def main():
    """CLI entry point: launch the FastAPI server with uvicorn."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

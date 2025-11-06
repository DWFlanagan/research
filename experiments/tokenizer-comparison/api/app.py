"""FastAPI application for tokenizer comparison."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compare_tokenizers import DEFAULT_PAPERS, DEFAULT_TOKENIZERS, run_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Tokenizer Comparison API",
    description="API for comparing tokenization algorithms on scientific papers",
    version="0.1.0",
)

# Job status storage (in-memory, would use Redis/DB in production)
jobs = {}


class RunRequest(BaseModel):
    """Request model for /run endpoint."""
    papers: Optional[list[dict]] = Field(
        default=None,
        description="List of paper dicts with 'name', 'url', 'arxiv_id'. Defaults to Attention, BERT, ByT5.",
    )
    tokenizers: Optional[list[str]] = Field(
        default=None,
        description="List of tokenizer keys. Defaults to all supported tokenizers.",
    )
    output_dir: Optional[str] = Field(
        default="output",
        description="Output directory path. Defaults to 'output'.",
    )


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Tokenizer Comparison API",
        "version": "0.1.0",
        "endpoints": {
            "/run": "POST - Run tokenizer comparison",
            "/status/{job_id}": "GET - Check job status",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/run")
async def run_experiment(request: RunRequest, background_tasks: BackgroundTasks):
    """
    Run tokenizer comparison experiment.
    
    This endpoint queues a comparison job and returns immediately with a job ID.
    Use /status/{job_id} to check progress.
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Use defaults if not provided
    papers = request.papers if request.papers else DEFAULT_PAPERS
    tokenizers = request.tokenizers if request.tokenizers else DEFAULT_TOKENIZERS
    output_dir = Path(request.output_dir)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "progress": "Job queued",
        "result": None,
        "error": None,
    }
    
    # Queue background task
    background_tasks.add_task(
        run_comparison_task,
        job_id,
        papers,
        tokenizers,
        output_dir,
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Experiment queued. Use /status/{job_id} to check progress.",
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a comparison job."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": f"Job {job_id} not found"},
        )
    
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.post("/run-sync")
async def run_experiment_sync(request: RunRequest):
    """
    Run tokenizer comparison experiment synchronously.
    
    This endpoint blocks until the comparison is complete.
    Use /run for async execution with job tracking.
    """
    papers = request.papers if request.papers else DEFAULT_PAPERS
    tokenizers = request.tokenizers if request.tokenizers else DEFAULT_TOKENIZERS
    output_dir = Path(request.output_dir)
    
    try:
        logger.info("Running synchronous comparison...")
        result = run_comparison(papers, tokenizers, output_dir)
        return {
            "status": "completed",
            "result": result,
        }
    except Exception as e:
        logger.error(f"Synchronous comparison failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": "An internal error occurred while running the comparison. Please check the server logs for details.",
            },
        )


async def run_comparison_task(
    job_id: str,
    papers: list[dict],
    tokenizers: list[str],
    output_dir: Path,
):
    """
    Background task to run comparison.
    
    Args:
        job_id: Job ID
        papers: List of papers
        tokenizers: List of tokenizers
        output_dir: Output directory
    """
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = "Starting experiment..."
        
        logger.info(f"Job {job_id}: Starting comparison")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_comparison,
            papers,
            tokenizers,
            output_dir,
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Experiment completed successfully"
        jobs[job_id]["result"] = result
        
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "An internal error occurred while running the comparison. Please check the server logs for details."


if __name__ == "__main__":
    import uvicorn
    
    # Use uvloop for better performance
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="uvloop",
    )

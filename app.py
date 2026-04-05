"""
Insurance Claim AI - FastAPI Backend Server
============================================
Runs on port 8000 by default.

Endpoints:
  POST /api/process-claim   - Process uploaded claim documents through agent pipeline
  GET  /api/health          - Health check
  GET  /api/agents          - List all agents and their status
  GET  /api/config          - Get current configuration
"""

import os
import sys
import json
import asyncio
import time
import logging
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging
from src.llm.gpt_client import GPTClient
from src.agents.image_agent import ImageProcessingAgent
from src.agents.pdf_agent import PDFProcessingAgent
from src.agents.requirements_agent import RequirementsAgent
from src.agents.credibility_agent import CredibilityAgent
from src.agents.billing_agent import BillingAgent
from src.agents.fraud_agent import FraudDetectionAgent
from src.agents.orchestrator import OrchestratorAgent
from src.handlers.error_handler import handle_agent_error

# ============================================================
# Setup
# ============================================================
setup_logging()
logger = logging.getLogger("insurance_claim_ai")

app = FastAPI(
    title="ClaimFlow AI",
    description="Multi-Agent Insurance Claim Processing System",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Initialize LLM Client and Agents
# ============================================================
API_KEY = os.getenv("OPENAI_API_KEY", "")
llm_client = None
if API_KEY:
    try:
        llm_client = GPTClient(api_key=API_KEY)
        logger.info("OpenAI GPT client initialized")
    except Exception as e:
        logger.warning(f"Could not initialize GPT client: {e}")

# Initialize all agents
agents = {
    "image": ImageProcessingAgent(llm_client=llm_client),
    "pdf": PDFProcessingAgent(llm_client=llm_client),
    "requirements": RequirementsAgent(llm_client=llm_client),
    "credibility": CredibilityAgent(llm_client=llm_client),
    "billing": BillingAgent(llm_client=llm_client),
    "fraud": FraudDetectionAgent(llm_client=llm_client),
}

orchestrator = OrchestratorAgent(llm_client=llm_client)

# Serve frontend static files if built
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ClaimFlow AI",
        "version": "1.0.0",
        "llm_connected": llm_client is not None,
        "agents_loaded": len(agents),
    }


@app.get("/api/agents")
async def list_agents():
    """List all registered agents."""
    agent_info = []
    for key, agent in agents.items():
        agent_info.append({
            "key": key,
            "name": agent.AGENT_NAME,
            "owner": agent.OWNER,
            "status": "active",
        })
    agent_info.append({
        "key": "orchestrator",
        "name": orchestrator.AGENT_NAME,
        "owner": orchestrator.OWNER,
        "status": "active",
    })
    return {"agents": agent_info}


@app.get("/api/config")
async def get_config():
    """Get current system configuration (non-sensitive)."""
    try:
        from config import get_model_config
        config = get_model_config()
        # Remove API keys
        for provider in ["openai", "claude"]:
            if provider in config:
                config[provider].pop("api_key_env", None)
        return {"config": config}
    except Exception:
        return {"config": "Config not loaded"}


@app.post("/api/process-claim")
async def process_claim(files: List[UploadFile] = File(...)):
    """
    Process uploaded claim documents through the full agent pipeline.

    Accepts: images (jpg, png) and PDFs of medical documents, policy documents, bills.
    Returns: Complete orchestrator assessment with all agent results.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    logger.info(f"Processing claim with {len(files)} file(s)")
    start_time = time.time()

    # Read first file (primary document)
    primary_file = files[0]
    file_bytes = await primary_file.read()
    filename = primary_file.filename or "document"
    mime_type = primary_file.content_type or "application/octet-stream"

    logger.info(f"Primary file: {filename} ({mime_type}, {len(file_bytes)} bytes)")

    try:
        # Run the orchestrator pipeline
        reasoning_log = []

        def on_log(msg):
            reasoning_log.append(msg)

        result = await orchestrator.run_pipeline(
            agents=agents,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            on_log=on_log,
        )

        elapsed = time.time() - start_time
        logger.info(f"Claim processed in {elapsed:.1f}s - Decision: {result.decision}")

        return JSONResponse(content={
            "success": True,
            "result": result.to_dict(),
            "total_time": f"{elapsed:.1f}s",
        })

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Serve frontend index.html for all non-API routes
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the React frontend."""
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return JSONResponse(content={"redirect": "Use frontend dev server at http://localhost:5173"})
    return JSONResponse(content={
        "message": "ClaimFlow AI Backend is running",
        "docs": "Visit /docs for API documentation",
        "health": "/api/health",
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting ClaimFlow AI backend on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )

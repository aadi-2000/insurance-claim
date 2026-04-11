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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
load_dotenv()

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


@app.post("/api/resume-claim")
async def resume_claim(files: List[UploadFile] = File(...), claim_id: str = Form(...)):
    """
    Resume a HOLD claim by uploading additional documents.
    
    This endpoint allows users to provide missing documents/information
    and continue processing from where the pipeline was halted.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    logger.info(f"Resuming claim {claim_id} with {len(files)} additional file(s)")
    start_time = time.time()
    
    # Process all uploaded files (could be multiple for missing fields)
    all_results = {"image": {}, "pdf": {}}
    
    for uploaded_file in files:
        file_bytes = await uploaded_file.read()
        filename = uploaded_file.filename or "document"
        mime_type = uploaded_file.content_type or "application/octet-stream"
        
        logger.info(f"Processing: {filename} ({mime_type}, {len(file_bytes)} bytes)")
        
        # Process based on file type
        if mime_type.startswith("image/"):
            result = await agents["image"].process(file_bytes, filename, mime_type)
            # Merge with existing image results
            if all_results["image"]:
                # Combine extracted text
                existing_text = all_results["image"].get("output", {}).get("extracted_text", "")
                new_text = result.get("output", {}).get("extracted_text", "")
                result["output"]["extracted_text"] = existing_text + "\n" + new_text
            all_results["image"] = result
            
        elif mime_type == "application/pdf":
            result = await agents["pdf"].process(file_bytes, filename)
            if all_results["pdf"]:
                existing_text = all_results["pdf"].get("output", {}).get("extracted_text", "")
                new_text = result.get("output", {}).get("extracted_text", "")
                result["output"]["extracted_text"] = existing_text + "\n" + new_text
            all_results["pdf"] = result
    
    try:
        # Re-run requirements validation with new documents
        requirements_result = await agents["requirements"].process(
            all_results.get("image", {}),
            all_results.get("pdf", {})
        )
        
        # Check if requirements are now met
        requirements_met = requirements_result.get("output", {}).get("requirements_met", False)
        missing_fields = requirements_result.get("output", {}).get("missing_fields", [])
        
        if not requirements_met:
            # Still missing fields - return HOLD again
            elapsed = time.time() - start_time
            logger.info(f"Claim still on HOLD - missing: {missing_fields}")
            
            return JSONResponse(content={
                "success": True,
                "status": "still_hold",
                "missing_fields": missing_fields,
                "message": "Additional documents received but some fields are still missing",
                "total_time": f"{elapsed:.1f}s",
            })
        
        # Requirements met - continue with full pipeline
        logger.info("Requirements now met - continuing with full pipeline")
        
        # Prepare combined results for remaining agents
        combined_results = {
            "image": all_results.get("image", {}),
            "pdf": all_results.get("pdf", {}),
            "requirements": requirements_result
        }
        
        # Run remaining agents (Credibility, Billing, Fraud)
        reasoning_log = []
        def on_log(msg):
            reasoning_log.append(msg)
        
        # Credibility Agent
        combined_results["credibility"] = await agents["credibility"].process(combined_results)
        
        # Billing Agent
        combined_results["billing"] = await agents["billing"].process(combined_results)
        
        # Fraud Agent
        combined_results["fraud"] = await agents["fraud"].process(combined_results)
        
        # Final decision fusion
        final_result = await orchestrator._fuse_decision(combined_results, reasoning_log)
        
        elapsed = time.time() - start_time
        logger.info(f"Resumed claim processed in {elapsed:.1f}s - Decision: {final_result.decision}")
        
        return JSONResponse(content={
            "success": True,
            "result": final_result.to_dict(),
            "total_time": f"{elapsed:.1f}s",
            "resumed": True,
        })
        
    except Exception as e:
        logger.error(f"Resume pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resume processing failed: {str(e)}")


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

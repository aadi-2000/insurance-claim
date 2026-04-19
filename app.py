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
from typing import List, Dict, Any

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
from src.utils.claim_history import ClaimHistoryDatabase
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

# In-memory session storage for resume functionality
# Maps claim_id -> {image_result, pdf_result, extracted_requirements}
claim_sessions = {}

app = FastAPI(
    title="ClaimFlow AI",
    description="Multi-Agent Insurance Claim Processing System",
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

# Initialize claim history database for fraud detection
logger.info("Initializing claim history database...")
claim_history_db = ClaimHistoryDatabase(storage_dir="data/claim_history")

# Initialize all agents
agents = {
    "image": ImageProcessingAgent(llm_client=llm_client),
    "pdf": PDFProcessingAgent(llm_client=llm_client),
    "requirements": RequirementsAgent(llm_client=llm_client),
    "credibility": CredibilityAgent(llm_client=llm_client),
    "billing": BillingAgent(llm_client=llm_client),
    "fraud": FraudDetectionAgent(llm_client=llm_client, claim_history_db=claim_history_db),
}

orchestrator = OrchestratorAgent(llm_client=llm_client, claim_history_db=claim_history_db)

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


@app.get("/api/claim-history/stats")
async def get_claim_history_stats():
    """Get statistics about the claim history database."""
    try:
        stats = claim_history_db.get_claim_statistics()
        return {
            "success": True,
            "statistics": stats,
        }
    except Exception as e:
        logger.error(f"Failed to get claim history stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@app.post("/api/approve-claim")
async def approve_claim(claim_data: Dict[str, Any]):
    """
    Manually approve a claim and store it in the RAG database.
    
    This endpoint allows manual approval of claims that may have been
    flagged for review, storing them in the claim history for future
    fraud detection.
    """
    try:
        # Extract claim information - handle both claim_summary structure
        claim_summary = claim_data.get("claim_summary", {})
        
        # Build extracted_data with all available fields
        # Map orchestrator field names to RAG database field names
        extracted_data = {
            "patient_name": claim_summary.get("patient_name") or claim_summary.get("patient"),
            "policy_number": claim_summary.get("policy_number") or claim_summary.get("policy") or claim_summary.get("patient_id"),
            "hospital_name": claim_summary.get("hospital_name") or claim_summary.get("hospital"),
            "diagnosis": claim_summary.get("diagnosis"),
            "admission_date": claim_summary.get("admission_date"),
            "discharge_date": claim_summary.get("discharge_date"),
            "total_claim_amount": claim_summary.get("total_claim_amount") or claim_summary.get("amount_claimed"),
        }
        
        # Remove None values
        extracted_data = {k: v for k, v in extracted_data.items() if v is not None}
        
        decision = "APPROVE"
        decision_reasons = claim_data.get("decision_reasons", ["Manually approved by reviewer"])
        
        # Validate required fields
        required_fields = ["patient_name", "policy_number", "hospital_name", "diagnosis"]
        missing = [f for f in required_fields if not extracted_data.get(f)]
        
        if missing:
            logger.error(f"Missing fields in approve request: {missing}")
            logger.error(f"Received claim_summary: {claim_summary}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required fields: {', '.join(missing)}"
            )
        
        # Store in claim history database
        claim_id = claim_history_db.add_claim(
            claim_data=extracted_data,
            decision=decision,
            decision_reasons=decision_reasons
        )
        
        logger.info(f"✅ Manually approved claim stored: {claim_id}")
        
        return {
            "success": True,
            "message": "Claim approved and stored in history database",
            "claim_id": claim_id,
            "decision": decision,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve claim: {e}")
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@app.post("/api/reject-claim")
async def reject_claim(claim_data: Dict[str, Any]):
    """
    Manually reject a claim and store it in the RAG database.
    
    This endpoint allows manual rejection of claims, storing them in the
    claim history for future fraud detection and pattern analysis.
    """
    try:
        # Extract claim information - handle both claim_summary structure
        claim_summary = claim_data.get("claim_summary", {})
        
        # Build extracted_data with all available fields
        # Map orchestrator field names to RAG database field names
        extracted_data = {
            "patient_name": claim_summary.get("patient_name") or claim_summary.get("patient"),
            "policy_number": claim_summary.get("policy_number") or claim_summary.get("policy") or claim_summary.get("patient_id"),
            "hospital_name": claim_summary.get("hospital_name") or claim_summary.get("hospital"),
            "diagnosis": claim_summary.get("diagnosis"),
            "admission_date": claim_summary.get("admission_date"),
            "discharge_date": claim_summary.get("discharge_date"),
            "total_claim_amount": claim_summary.get("total_claim_amount") or claim_summary.get("amount_claimed"),
        }
        
        # Remove None values
        extracted_data = {k: v for k, v in extracted_data.items() if v is not None}
        
        decision = "REJECT"
        decision_reasons = claim_data.get("decision_reasons", ["Manually rejected by reviewer"])
        
        # Validate required fields
        required_fields = ["patient_name", "policy_number", "hospital_name", "diagnosis"]
        missing = [f for f in required_fields if not extracted_data.get(f)]
        
        if missing:
            logger.error(f"Missing fields in reject request: {missing}")
            logger.error(f"Received claim_summary: {claim_summary}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required fields: {', '.join(missing)}"
            )
        
        # Store in claim history database
        claim_id = claim_history_db.add_claim(
            claim_data=extracted_data,
            decision=decision,
            decision_reasons=decision_reasons
        )
        
        logger.info(f"❌ Manually rejected claim stored: {claim_id}")
        
        return {
            "success": True,
            "message": "Claim rejected and stored in history database",
            "claim_id": claim_id,
            "decision": decision,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject claim: {e}")
        raise HTTPException(status_code=500, detail=f"Rejection failed: {str(e)}")


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

    try:
        # Process all uploaded files
        all_results = {"image": {}, "pdf": {}}
        
        for uploaded_file in files:
            file_bytes = await uploaded_file.read()
            filename = uploaded_file.filename or "document"
            mime_type = uploaded_file.content_type or "application/octet-stream"
            
            logger.info(f"Processing file: {filename} ({mime_type}, {len(file_bytes)} bytes)")
            
            # Process based on file type
            if mime_type.startswith("image/"):
                result = await agents["image"].process(file_bytes, filename, mime_type)
                # Merge with existing image results
                if all_results["image"] and all_results["image"].get("output"):
                    existing_text = all_results["image"].get("output", {}).get("extracted_text", "")
                    new_text = result.get("output", {}).get("extracted_text", "")
                    result["output"]["extracted_text"] = existing_text + "\n\n" + new_text
                all_results["image"] = result
                
            elif mime_type == "application/pdf":
                result = await agents["pdf"].process(file_bytes, filename)
                # Merge with existing PDF results
                if all_results["pdf"] and all_results["pdf"].get("output"):
                    existing_text = all_results["pdf"].get("output", {}).get("extracted_text", "")
                    new_text = result.get("output", {}).get("extracted_text", "")
                    result["output"]["extracted_text"] = existing_text + "\n\n" + new_text
                all_results["pdf"] = result
        
        # Run the orchestrator pipeline with combined results
        reasoning_log = []

        def on_log(msg):
            reasoning_log.append(msg)

        result = await orchestrator.run_pipeline_with_results(
            agents=agents,
            image_result=all_results.get("image", {}),
            pdf_result=all_results.get("pdf", {}),
            on_log=on_log,
        )

        elapsed = time.time() - start_time
        logger.info(f"Claim processed in {elapsed:.1f}s - Decision: {result.decision}")

        # Store session data for resume functionality if claim is on HOLD
        if result.status == "pending_documents":
            claim_id = result.claim_summary.get("claim_id", f"claim_{int(time.time())}")
            claim_sessions[claim_id] = {
                "image_result": all_results.get("image", {}),
                "pdf_result": all_results.get("pdf", {}),
                "extracted_requirements": result.claim_summary,
                "timestamp": time.time()
            }
            logger.info(f"✅ Stored session data for claim {claim_id}")
            logger.info(f"   Extracted fields: {list(result.claim_summary.keys())}")
            logger.info(f"   Diagnosis value: {result.claim_summary.get('diagnosis')}")

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
    
    # Retrieve previous session data
    previous_session = claim_sessions.get(claim_id, {})
    if previous_session:
        logger.info(f"✅ Found previous session data for claim {claim_id}")
        logger.info(f"   Previous diagnosis: {previous_session.get('extracted_requirements', {}).get('diagnosis')}")
    else:
        logger.warning(f"⚠️ No previous session found for claim {claim_id}")
        logger.info(f"   Available sessions: {list(claim_sessions.keys())}")
    
    # Start with previous results or empty
    all_results = {
        "image": previous_session.get("image_result", {}),
        "pdf": previous_session.get("pdf_result", {})
    }
    
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
        
        # Merge with previous extracted requirements to preserve fields from first upload
        previous_extracted = previous_session.get("extracted_requirements", {})
        newly_extracted = requirements_result.get("output", {}).get("extracted_requirements", {})
        
        # Merge: keep previous values, add new values, prefer new if both exist
        merged_requirements = {}
        for field in ["patient_name", "policy_number", "hospital_name", "diagnosis", 
                      "admission_date", "discharge_date", "total_claim_amount"]:
            # Use new value if present, otherwise use previous value
            if newly_extracted.get(field):
                merged_requirements[field] = newly_extracted[field]
            elif previous_extracted.get(field):
                merged_requirements[field] = previous_extracted[field]
            else:
                merged_requirements[field] = None
        
        # Update requirements result with merged data
        requirements_result["output"]["extracted_requirements"] = merged_requirements
        
        # Recalculate missing fields based on merged requirements
        missing_fields = [f for f in ["patient_name", "policy_number", "hospital_name", "diagnosis",
                                       "admission_date", "discharge_date", "total_claim_amount"]
                          if not merged_requirements.get(f)]
        requirements_met = len(missing_fields) == 0
        requirements_result["output"]["requirements_met"] = requirements_met
        requirements_result["output"]["missing_fields"] = missing_fields
        
        if not requirements_met:
            # Still missing fields - update session and return HOLD again
            elapsed = time.time() - start_time
            logger.info(f"Claim still on HOLD - missing: {missing_fields}")
            
            # Update session with merged data
            extracted_reqs = requirements_result.get("output", {}).get("extracted_requirements", {})
            claim_sessions[claim_id] = {
                "image_result": all_results.get("image", {}),
                "pdf_result": all_results.get("pdf", {}),
                "extracted_requirements": extracted_reqs,
                "timestamp": time.time()
            }
            
            return JSONResponse(content={
                "success": True,
                "status": "still_hold",
                "missing_fields": missing_fields,
                "extracted_fields": extracted_reqs,
                "message": "Additional documents received but some fields are still missing",
                "total_time": f"{elapsed:.1f}s",
            })
        
        # Requirements met - continue with full pipeline using orchestrator
        logger.info("Requirements now met - continuing with full pipeline")
        
        # Use orchestrator's _continue_pipeline to run remaining agents
        reasoning_log = []
        def on_log(msg):
            reasoning_log.append(msg)
        
        # Build results dict with image, pdf, and requirements
        results = {
            "image": all_results.get("image", {}),
            "pdf": all_results.get("pdf", {}),
            "requirements": requirements_result
        }
        
        # Continue pipeline from Credibility agent onwards
        # This will run Credibility, Billing, Fraud, and do final decision fusion
        final_result = await orchestrator._continue_pipeline(
            agents=agents,
            results=results,
            log_messages=reasoning_log,
            start_time=start_time,
            log=on_log
        )
        
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

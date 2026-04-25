"""
Orchestration & Integration Agent (Agent 4) - Master Orchestrator
Owner: Aadithya Pabbisetty
Status: FULLY IMPLEMENTED

This is the central coordinator that:
  1. Triggers all 6 specialized agents in the correct pipeline order
  2. Collects and validates results from each agent
  3. Applies weighted decision fusion logic
  4. Determines final claim decision (APPROVE / REJECT / REVIEW / HOLD)
  5. Uses LLM to generate natural language summary
  6. Produces comprehensive assessment report with reasoning traces

Architecture:
  - Sequential pipeline with dependency management
  - Weighted confidence scoring across agents
  - Multi-threshold decision logic (fraud, credibility, billing, requirements)
  - LLM-powered natural language report generation
  - Full reasoning trace logging for auditability
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("insurance_claim_ai.agents")

# Import SSE manager for real-time updates
try:
    from src.utils.sse_manager import sse_manager
except ImportError:
    sse_manager = None


# ============================================================
# Data Classes for Type Safety
# ============================================================

@dataclass
class AgentWeight:
    """Weight configuration for an agent in decision fusion."""
    agent_key: str
    weight: float
    critical: bool = False  # If True, failure forces REJECT


@dataclass
class DecisionThresholds:
    """Thresholds for the orchestrator's decision logic."""
    fraud_reject: float = 0.5
    fraud_review: float = 0.3
    credibility_reject: float = 0.4
    billing_anomaly_flag: float = 0.6
    minimum_confidence: float = 0.5


@dataclass
class ReasoningStep:
    """A single step in the orchestrator's reasoning trace."""
    timestamp: float
    step: str
    detail: str
    agent: Optional[str] = None

    def to_log_string(self) -> str:
        prefix = f"[{self.agent}] " if self.agent else ""
        return f"{prefix}{self.step}: {self.detail}"


@dataclass
class OrchestratorResult:
    """Complete result from the orchestrator."""
    agent: str = "Master Orchestrator"
    owner: str = "Aadithya Pabbisetty"
    status: str = "success"
    decision: str = ""
    decision_reasons: List[str] = field(default_factory=list)
    weighted_confidence: float = 0.0
    agent_summaries: List[Dict] = field(default_factory=list)
    claim_summary: Dict = field(default_factory=dict)
    processing_summary: Dict = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# Orchestrator Agent
# ============================================================

class OrchestratorAgent:
    """
    Master Orchestrator Agent - Coordinates all specialized agents
    and applies decision fusion logic for final claim assessment.
    """

    AGENT_NAME = "Master Orchestrator"
    OWNER = "Aadithya Pabbisetty"
    
    # Default agent weights for decision fusion
    DEFAULT_WEIGHTS = [
        AgentWeight("image", 0.15),
        AgentWeight("pdf", 0.10),
        AgentWeight("requirements", 0.20, critical=True),
        AgentWeight("credibility", 0.20),
        AgentWeight("billing", 0.15),
        AgentWeight("fraud", 0.2, critical=True),
    ]

    def __init__(self, llm_client=None, weights: Optional[List[AgentWeight]] = None,
                 thresholds: Optional[DecisionThresholds] = None, claim_history_db=None):
        """
        Initialize the Orchestrator.

        Args:
            llm_client: LLM client for generating natural language summaries
            weights: Custom agent weights (uses defaults if None)
            thresholds: Custom decision thresholds (uses defaults if None)
            claim_history_db: ClaimHistoryDatabase instance for storing processed claims
        """
        self.llm_client = llm_client
        self.weights = {w.agent_key: w for w in (weights or self.DEFAULT_WEIGHTS)}
        self.thresholds = thresholds or DecisionThresholds()
        self.claim_history_db = claim_history_db
        self.reasoning_trace: List[ReasoningStep] = []
        logger.info(f"[{self.AGENT_NAME}] Initialized with {len(self.weights)} agent weights")

    def _log_step(self, step: str, detail: str, agent: Optional[str] = None) -> str:
        """Log a reasoning step and return formatted string."""
        entry = ReasoningStep(
            timestamp=time.time(),
            step=step,
            detail=detail,
            agent=agent,
        )
        self.reasoning_trace.append(entry)
        log_str = entry.to_log_string()
        logger.debug(f"[Orchestrator] {log_str}")
        return log_str

    async def process_multiple_files(self, agents: Dict[str, Any], files: List[Dict[str, Any]],
                                      output_dir: str = "data/outputs") -> Dict[str, Any]:
        """
        Process multiple files (images and PDFs) and gather all extracted data.
        
        Args:
            agents: Dict of agent_key -> agent_instance
            files: List of file dicts with 'bytes', 'filename', 'mime_type'
            output_dir: Directory to save output files
            
        Returns:
            Dict with all extracted data and processing results
        """
        
        all_extracted_data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "files_processed": [],
            "image_extractions": [],
            "pdf_extractions": [],
            "combined_text": ""
        }
        
        # Process each file
        for idx, file_info in enumerate(files, 1):
            file_bytes = file_info['bytes']
            filename = file_info['filename']
            mime_type = file_info.get('mime_type', 'application/octet-stream')
            
            
            file_result = {
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": len(file_bytes),
                "processing_status": "pending"
            }
            
            # Determine which agent to use based on file type
            if 'image' in mime_type or filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                try:
                    result = await agents["image"].process(file_bytes, filename, mime_type)
                    extracted_text = result.get('output', {}).get('extracted_text', '')
                    
                    extraction_data = {
                        "file_index": idx,
                        "filename": filename,
                        "agent": "Image Processing Agent",
                        "extracted_text": extracted_text,
                        "text_length": len(extracted_text),
                        "metadata": result.get('output', {}).get('image_metadata', {}),
                        "extraction_method": result.get('output', {}).get('extraction_method', 'Unknown'),
                        "confidence": result.get('confidence', 0)
                    }
                    
                    all_extracted_data["image_extractions"].append(extraction_data)
                    all_extracted_data["combined_text"] += f"\n\n--- {filename} (Image) ---\n{extracted_text}"
                    
                    file_result["processing_status"] = "success"
                    file_result["agent_used"] = "image"
                    file_result["text_extracted"] = len(extracted_text)
                    
                    
                except Exception as e:
                    file_result["processing_status"] = "error"
                    file_result["error"] = str(e)
                    
            elif 'pdf' in mime_type or filename.lower().endswith('.pdf'):
                try:
                    result = await agents["pdf"].process(file_bytes, filename)
                    extracted_text = result.get('output', {}).get('extracted_text', '')
                    sections = result.get('output', {}).get('sections', [])
                    
                    extraction_data = {
                        "file_index": idx,
                        "filename": filename,
                        "agent": "PDF Processing Agent",
                        "extracted_text": extracted_text,
                        "text_length": len(extracted_text),
                        "num_pages": result.get('output', {}).get('num_pages', 0),
                        "num_sections": len(sections),
                        "sections": sections,
                        "metadata": result.get('output', {}).get('pdf_metadata', {}),
                        "confidence": result.get('confidence', 0)
                    }
                    
                    all_extracted_data["pdf_extractions"].append(extraction_data)
                    all_extracted_data["combined_text"] += f"\n\n--- {filename} (PDF) ---\n{extracted_text}"
                    
                    file_result["processing_status"] = "success"
                    file_result["agent_used"] = "pdf"
                    file_result["text_extracted"] = len(extracted_text)
                    file_result["pages"] = extraction_data["num_pages"]
                    
                    
                except Exception as e:
                    file_result["processing_status"] = "error"
                    file_result["error"] = str(e)
            else:
                file_result["processing_status"] = "skipped"
                file_result["reason"] = "Unsupported file type"
            
            all_extracted_data["files_processed"].append(file_result)
        
        # Save to output file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"extraction_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INSURANCE CLAIM AI - DOCUMENT EXTRACTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {all_extracted_data['timestamp']}\n")
            f.write(f"Total Files Processed: {all_extracted_data['total_files']}\n\n")
            
            # Summary
            f.write("FILE PROCESSING SUMMARY\n")
            f.write("-" * 80 + "\n")
            for file_info in all_extracted_data['files_processed']:
                f.write(f"\nFile: {file_info['filename']}\n")
                f.write(f"  Status: {file_info['processing_status']}\n")
                if file_info['processing_status'] == 'success':
                    f.write(f"  Agent: {file_info['agent_used']}\n")
                    f.write(f"  Text Extracted: {file_info.get('text_extracted', 0)} characters\n")
                    if 'pages' in file_info:
                        f.write(f"  Pages: {file_info['pages']}\n")
            
            # Image extractions
            if all_extracted_data['image_extractions']:
                f.write("\n\n" + "="*80 + "\n")
                f.write("IMAGE EXTRACTIONS\n")
                f.write("="*80 + "\n")
                for img_data in all_extracted_data['image_extractions']:
                    f.write(f"\n[{img_data['file_index']}] {img_data['filename']}\n")
                    f.write(f"Method: {img_data['extraction_method']}\n")
                    f.write(f"Confidence: {img_data['confidence']:.2f}\n")
                    f.write(f"Text Length: {img_data['text_length']} characters\n")
                    f.write(f"\nExtracted Text:\n{'-'*80}\n")
                    f.write(img_data['extracted_text'] or "(No text extracted)")
                    f.write("\n" + "-"*80 + "\n")
            
            # PDF extractions
            if all_extracted_data['pdf_extractions']:
                f.write("\n\n" + "="*80 + "\n")
                f.write("PDF EXTRACTIONS\n")
                f.write("="*80 + "\n")
                for pdf_data in all_extracted_data['pdf_extractions']:
                    f.write(f"\n[{pdf_data['file_index']}] {pdf_data['filename']}\n")
                    f.write(f"Pages: {pdf_data['num_pages']}\n")
                    f.write(f"Sections: {pdf_data['num_sections']}\n")
                    f.write(f"Confidence: {pdf_data['confidence']:.2f}\n")
                    f.write(f"Text Length: {pdf_data['text_length']} characters\n")
                    
                    if pdf_data['sections']:
                        f.write(f"\nSections Found:\n")
                        for section in pdf_data['sections']:
                            f.write(f"  - {section.get('title', 'Untitled')}\n")
                    
                    f.write(f"\nExtracted Text:\n{'-'*80}\n")
                    f.write(pdf_data['extracted_text'] or "(No text extracted)")
                    f.write("\n" + "-"*80 + "\n")
            
            # Combined text
            f.write("\n\n" + "="*80 + "\n")
            f.write("COMBINED EXTRACTED TEXT (ALL FILES)\n")
            f.write("="*80 + "\n")
            f.write(all_extracted_data['combined_text'])
        
        
        all_extracted_data["output_file"] = str(output_file)
        return all_extracted_data

    async def run_pipeline_with_results(
        self,
        agents: Dict[str, Any],
        image_result: Dict[str, Any],
        pdf_result: Dict[str, Any],
        on_log=None,
        claim_id: Optional[str] = None,
    ) -> "OrchestratorResult":
        """
        Run orchestrator pipeline with pre-processed image and PDF results.
        Used when multiple files are uploaded and need to be combined.
        
        Args:
            agents: Dict of agent_key -> agent_instance
            image_result: Pre-processed image agent result
            pdf_result: Pre-processed PDF agent result
            on_log: Optional callback for real-time log streaming
            
        Returns:
            OrchestratorResult with complete assessment
        """
        self.reasoning_trace = []
        start_time = time.time()
        results = {}
        log_messages = []

        def log(msg):
            log_messages.append(msg)
            if on_log:
                on_log(msg)

        log(self._log_step("INIT", "Orchestrator Agent initialized - coordinating all agent outputs"))

        # Use pre-processed results or mark as skipped
        if image_result and image_result.get("output"):
            results["image"] = image_result
            log(self._log_step("RESULT", f"Image agent results loaded: conf={image_result.get('confidence', 0)}", "image"))
        else:
            results["image"] = {"status": "skipped", "reason": "No image files uploaded"}
            log(self._log_step("SKIP", "Image agent skipped - no image files", "image"))
        
        if pdf_result and pdf_result.get("output"):
            results["pdf"] = pdf_result
            log(self._log_step("RESULT", f"PDF agent results loaded: conf={pdf_result.get('confidence', 0)}", "pdf"))
        else:
            results["pdf"] = {"status": "skipped", "reason": "No PDF files uploaded"}
            log(self._log_step("SKIP", "PDF agent skipped - no PDF files", "pdf"))

        # Continue with Requirements Agent and rest of pipeline
        return await self._continue_pipeline(agents, results, log_messages, start_time, log, claim_id)

    async def _continue_pipeline(self, agents, results, log_messages, start_time, log, claim_id=None):
        """Continue pipeline from Requirements Agent onwards"""
        # ---- Step 3: Requirements Validation ----
        log(self._log_step("PIPELINE", "Starting Requirements Agent", "requirements"))
        
        # Send PROCESSING status
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "PROCESSING")
        
        try:
            results["requirements"] = await agents["requirements"].process(
                results.get("image", {}), results.get("pdf", {})
            )
            log(self._log_step("RESULT", f"Requirements agent completed: conf={results['requirements']['confidence']}", "requirements"))
            
            # Send SSE event for Requirements agent completion
            if sse_manager and claim_id:
                requirements_met = results["requirements"].get("output", {}).get("requirements_met", False)
                missing_fields = results["requirements"].get("output", {}).get("missing_fields", [])
                
                # Send SSE event for Requirements agent - WARNING if requirements not met
                if sse_manager and claim_id:
                    if requirements_met:
                        await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "COMPLETED", results["requirements"]["confidence"])
                    else:
                        await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "WARNING", results["requirements"]["confidence"])
                
                # Check if requirements are met
            
            # Check if requirements are met
            requirements_met = results["requirements"].get("output", {}).get("requirements_met", False)
            missing_fields = results["requirements"].get("output", {}).get("missing_fields", [])
            if not requirements_met and missing_fields:
                log(self._log_step("HOLD", f"Missing required documents/fields: {', '.join(missing_fields)}", "requirements"))
                
                # Return early with request for missing documents
                elapsed = time.time() - start_time
                extracted_reqs = results["requirements"].get("output", {}).get("extracted_requirements", {})
                
                # Generate unique claim ID for session tracking
                claim_id = f"claim_{int(time.time() * 1000)}"
                
                return OrchestratorResult(
                    status="pending_documents",
                    decision="HOLD",
                    decision_reasons=[
                        f"Missing {len(missing_fields)} required field(s)",
                        "User action required: Please provide missing documents"
                    ],
                    weighted_confidence=0.0,
                    agent_summaries=[{
                        "agent": "Requirements Agent",
                        "status": "INCOMPLETE",
                        "missing_fields": missing_fields
                    }],
                    claim_summary={
                        "claim_id": claim_id,
                        "status": "PENDING_DOCUMENTS",
                        "missing_fields": missing_fields,
                        "action_required": "Please upload documents containing the missing information",
                        "patient_name": extracted_reqs.get("patient_name"),
                        "policy_number": extracted_reqs.get("policy_number"),
                        "hospital_name": extracted_reqs.get("hospital_name"),
                        "diagnosis": extracted_reqs.get("diagnosis"),
                        "admission_date": extracted_reqs.get("admission_date"),
                        "discharge_date": extracted_reqs.get("discharge_date"),
                        "total_claim_amount": extracted_reqs.get("total_claim_amount"),
                    },
                    processing_summary={
                        "total_agents": 3,
                        "agents_passed": 2,
                        "orchestrator_time": f"{elapsed:.1f}s",
                        "halt_reason": "Missing required documents"
                    },
                    reasoning_trace=log_messages,
                )
                
        except Exception as e:
            log(self._log_step("ERROR", f"Requirements agent failed: {e}", "requirements"))
            results["requirements"] = self._create_error_result("Requirements Agent", "Karthikeyan Pillai", str(e))
            
            # Send FAILED status
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "FAILED")

        # ---- Steps 4-6: Parallel Execution (Credibility, Billing, Fraud) ----
        log(self._log_step("PIPELINE", "Starting parallel agents: Credibility, Billing, Fraud"))
        
        # Send PROCESSING status for all three agents simultaneously
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Credibility & Policy Interpretation Agent", "PROCESSING")
            await sse_manager.send_agent_status(claim_id, "Billing & Processing Agent", "PROCESSING")
            await sse_manager.send_agent_status(claim_id, "AI Fraud Detection Agent", "PROCESSING")
        
        # Execute all three agents in parallel
        async def run_credibility():
            try:
                result = await agents["credibility"].process(results)
                log(self._log_step("RESULT", f"Credibility agent completed: conf={result['confidence']}", "credibility"))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Credibility & Policy Interpretation Agent", "COMPLETED", result['confidence'])
                return result
            except Exception as e:
                log(self._log_step("ERROR", f"Credibility agent failed: {e}", "credibility"))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Credibility & Policy Interpretation Agent", "FAILED")
                return self._create_error_result("Credibility Agent", "Shruti Roy", str(e))
        
        async def run_billing():
            try:
                result = await agents["billing"].process(results)
                log(self._log_step("RESULT", f"Billing agent completed: conf={result['confidence']}", "billing"))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Billing & Processing Agent", "COMPLETED", result['confidence'])
                return result
            except Exception as e:
                log(self._log_step("ERROR", f"Billing agent failed: {e}", "billing"))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Billing & Processing Agent", "FAILED")
                return self._create_error_result("Billing Agent", "Siri Spandana", str(e))
        
        async def run_fraud():
            try:
                result = await agents["fraud"].process(results)
                log(self._log_step("RESULT", f"Fraud agent completed: conf={result['confidence']}", "fraud"))
                if sse_manager and claim_id:
                    fraud_category = result.get('output', {}).get('fraud_category', 'NONE')
                    await sse_manager.send_agent_status(claim_id, "AI Fraud Detection Agent", "COMPLETED", fraud_category)
                return result
            except Exception as e:
                log(self._log_step("ERROR", f"Fraud agent failed: {e}", "fraud"))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "AI Fraud Detection Agent", "FAILED")
                return self._create_error_result("Fraud Detection Agent", "Titash Bhattacharya", str(e))
        
        # Run all three in parallel
        credibility_result, billing_result, fraud_result = await asyncio.gather(
            run_credibility(),
            run_billing(),
            run_fraud()
        )
        
        results["credibility"] = credibility_result
        results["billing"] = billing_result
        results["fraud"] = fraud_result
        
        # Check credibility score after parallel execution
        credibility_score = results["credibility"].get("output", {}).get("credibility_score", 1.0)
        if credibility_score < self.thresholds.credibility_reject:
            log(self._log_step("REJECT", f"Low credibility score: {credibility_score:.2f} < {self.thresholds.credibility_reject}", "credibility"))
            elapsed = time.time() - start_time
            return OrchestratorResult(
                status="rejected",
                decision="REJECT",
                decision_reasons=[
                    f"Low credibility score: {credibility_score:.2f}",
                    f"Score below minimum threshold of {self.thresholds.credibility_reject}"
                ],
                weighted_confidence=credibility_score,
                agent_summaries=[{"agent": "Credibility Agent", "status": "FAILED", "score": credibility_score}],
                claim_summary={"status": "REJECTED", "credibility_score": credibility_score},
                processing_summary={"total_agents": 4, "agents_passed": 3, "orchestrator_time": f"{elapsed:.1f}s"},
                reasoning_trace=log_messages,
            )

        # ---- Step 7: Decision Fusion ----
        log(self._log_step("FUSION", "Applying weighted decision fusion logic"))
        
        # Send Orchestrator PROCESSING status when doing business logic
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Master Orchestrator", "PROCESSING")
        
        decision_result = self._compute_decision(results)
        log(self._log_step("DECISION", f"Final decision: {decision_result['decision']}"))

        # ---- Step 8: LLM Summary ----
        ai_summary = ""
        if self.llm_client:
            try:
                ai_summary = await self._generate_summary(decision_result)
            except:
                pass

        # ---- Build Final Result ----
        elapsed = time.time() - start_time
        
        # Send Orchestrator COMPLETED status
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Master Orchestrator", "COMPLETED", decision_result["weighted_confidence"])
        
        return OrchestratorResult(
            status="success",
            decision=decision_result["decision"],
            decision_reasons=decision_result["reasons"],
            weighted_confidence=decision_result["weighted_confidence"],
            agent_summaries=decision_result["agent_summaries"],
            claim_summary=self._build_claim_summary(results),
            processing_summary={
                "total_agents": len(results),
                "agents_passed": sum(1 for r in results.values() if r.get("status") == "success"),
                "orchestrator_time": f"{elapsed:.1f}s",
                "ai_summary": ai_summary,
            },
            reasoning_trace=log_messages,
        )

    async def run_pipeline(self, agents: Dict[str, Any], file_bytes: bytes,
                            filename: str, mime_type: str = "image/jpeg",
                            on_log: Optional[Callable] = None, claim_id: Optional[str] = None) -> OrchestratorResult:
        """
        Execute the full agent pipeline and produce a final decision.

        Args:
            agents: Dict of agent_key -> agent_instance
            file_bytes: Raw uploaded file bytes
            filename: Original filename
            mime_type: File MIME type
            on_log: Optional callback for real-time log streaming

        Returns:
            OrchestratorResult with complete assessment
        """
        self.reasoning_trace = []
        start_time = time.time()
        results = {}
        log_messages = []

        def log(msg):
            log_messages.append(msg)
            if on_log:
                on_log(msg)

        log(self._log_step("INIT", "Orchestrator Agent initialized - coordinating all agent outputs"))

        # ---- Step 1: Determine file type and process accordingly ----
        is_image = 'image' in mime_type or filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        is_pdf = 'pdf' in mime_type or filename.lower().endswith('.pdf')
        
        # Process Image if applicable
        if is_image:
            log(self._log_step("PIPELINE", "Starting Image Processing Agent", "image"))
            
            # Send PROCESSING status
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "Image Processing Agent", "PROCESSING")
            
            try:
                results["image"] = await agents["image"].process(file_bytes, filename, mime_type)
                log(self._log_step("RESULT", f"Image agent completed: conf={results['image']['confidence']}", "image"))
                
                # Emit SSE event
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Image Processing Agent", "COMPLETED", results["image"]['confidence'])
                
                # Print extracted data to console
                extracted_text = results["image"].get('output', {}).get('extracted_text', '')
            except Exception as e:
                log(self._log_step("ERROR", f"Image agent failed: {e}", "image"))
                results["image"] = self._create_error_result("Image Processing Agent", "Vivek Vardhan", str(e))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "Image Processing Agent", "FAILED")
        else:
            results["image"] = {"status": "skipped", "reason": "Not an image file"}
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "Image Processing Agent", "SKIPPED")

        # Process PDF if applicable
        if is_pdf:
            log(self._log_step("PIPELINE", "Starting PDF Processing Agent", "pdf"))
            
            # Send PROCESSING status
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "PDF Processing Agent", "PROCESSING")
            
            try:
                results["pdf"] = await agents["pdf"].process(file_bytes, filename)
                log(self._log_step("RESULT", f"PDF agent completed: conf={results['pdf']['confidence']}", "pdf"))
                
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "PDF Processing Agent", "COMPLETED", results["pdf"]['confidence'])
                
            except Exception as e:
                log(self._log_step("ERROR", f"PDF agent failed: {e}", "pdf"))
                results["pdf"] = self._create_error_result("PDF Processing Agent", "Swapnil Sontakke", str(e))
                if sse_manager and claim_id:
                    await sse_manager.send_agent_status(claim_id, "PDF Processing Agent", "FAILED")
        else:
            results["pdf"] = {"status": "skipped", "reason": "Not a PDF file"}
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "PDF Processing Agent", "SKIPPED")

        # ---- Step 3: Requirements Validation ----
        log(self._log_step("PIPELINE", "Starting Requirements Agent", "requirements"))
        
        # Send PROCESSING status
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "PROCESSING")
        
        try:
            results["requirements"] = await agents["requirements"].process(
                results.get("image", {}), results.get("pdf", {})
            )
            log(self._log_step("RESULT", f"Requirements agent completed: conf={results['requirements']['confidence']}", "requirements"))
            
            # Check if requirements are met
            requirements_met = results["requirements"].get("output", {}).get("requirements_met", False)
            missing_fields = results["requirements"].get("output", {}).get("missing_fields", [])
            
            # Send SSE event for Requirements agent - WARNING if requirements not met
            if sse_manager and claim_id:
                if requirements_met:
                    await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "COMPLETED", results["requirements"]['confidence'])
                else:
                    await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "WARNING", results["requirements"]['confidence'])
            
            if not requirements_met and missing_fields:
                log(self._log_step("HOLD", f"Missing required documents/fields: {', '.join(missing_fields)}", "requirements"))
                
                # Return early with request for missing documents
                elapsed = time.time() - start_time
                # Get extracted requirements to show what was found
                extracted_reqs = results["requirements"].get("output", {}).get("extracted_requirements", {})
                
                return OrchestratorResult(
                    status="pending_documents",
                    decision="HOLD",
                    decision_reasons=[
                        f"Missing {len(missing_fields)} required field(s)",
                        "User action required: Please provide missing documents"
                    ],
                    weighted_confidence=0.0,
                    agent_summaries=[{
                        "agent": "Requirements Agent",
                        "status": "INCOMPLETE",
                        "missing_fields": missing_fields
                    }],
                    claim_summary={
                        "status": "PENDING_DOCUMENTS",
                        "missing_fields": missing_fields,
                        "action_required": "Please upload documents containing the missing information",
                        "patient_name": extracted_reqs.get("patient_name"),
                        "policy_number": extracted_reqs.get("policy_number"),
                        "hospital_name": extracted_reqs.get("hospital_name"),
                        "diagnosis": extracted_reqs.get("diagnosis"),
                        "admission_date": extracted_reqs.get("admission_date"),
                        "discharge_date": extracted_reqs.get("discharge_date"),
                        "total_claim_amount": extracted_reqs.get("total_claim_amount"),
                    },
                    processing_summary={
                        "total_agents": 3,
                        "agents_passed": 2,
                        "orchestrator_time": f"{elapsed:.1f}s",
                        "halt_reason": "Missing required documents"
                    },
                    reasoning_trace=log_messages,
                )
                
        except Exception as e:
            log(self._log_step("ERROR", f"Requirements agent failed: {e}", "requirements"))
            results["requirements"] = self._create_error_result("Requirements Agent", "Karthikeyan Pillai", str(e))
            
            # Send FAILED status
            if sse_manager and claim_id:
                await sse_manager.send_agent_status(claim_id, "Requirements & Document Validation Agent", "FAILED")

        # ---- Step 4: Credibility & Policy ----
        log(self._log_step("PIPELINE", "Starting Credibility Agent", "credibility"))
        try:
            results["credibility"] = await agents["credibility"].process(results)
            log(self._log_step("RESULT", f"Credibility agent completed: conf={results['credibility']['confidence']}", "credibility"))

            credibility_score = results["credibility"].get("output", {}).get("credibility_score", 1.0)

            if credibility_score < self.thresholds.credibility_reject:
                log(self._log_step("REJECT", f"Low credibility score: {credibility_score:.2f} < {self.thresholds.credibility_reject}", "credibility"))

                elapsed = time.time() - start_time
                return OrchestratorResult(
                    status="rejected",
                    decision="REJECT",
                    decision_reasons=[
                        f"Low credibility score: {credibility_score:.2f}",
                        f"Score below minimum threshold of {self.thresholds.credibility_reject}"
                    ],
                    weighted_confidence=credibility_score,
                    agent_summaries=[
                        {"agent": "Image Processing Agent", "status": "COMPLETED"},
                        {"agent": "PDF Processing Agent", "status": "COMPLETED"},
                        {"agent": "Requirements Agent", "status": "COMPLETED"},
                        {"agent": "Credibility Agent", "status": "FAILED", "score": credibility_score}
                    ],
                    claim_summary={
                        "status": "REJECTED",
                        "credibility_score": credibility_score,
                        "rejection_reason": "Low user credibility"
                    },
                    processing_summary={
                        "total_agents": 4,
                        "agents_passed": 3,
                        "orchestrator_time": f"{elapsed:.1f}s",
                        "halt_reason": "Credibility check failed"
                    },
                    reasoning_trace=log_messages,
                )

        except Exception as e:
            log(self._log_step("ERROR", f"Credibility agent failed: {e}", "credibility"))
            results["credibility"] = self._create_error_result("Credibility Agent", "Shruti Roy", str(e))

        # ---- Step 5: Billing ----
        log(self._log_step("PIPELINE", "Starting Billing Agent", "billing"))
        try:
            results["billing"] = await agents["billing"].process(results)
            log(self._log_step("RESULT", f"Billing agent completed: conf={results['billing']['confidence']}", "billing"))
        except Exception as e:
            log(self._log_step("ERROR", f"Billing agent failed: {e}", "billing"))
            results["billing"] = self._create_error_result("Billing Agent", "Siri Spandana", str(e))

        # ---- Step 6: Fraud Detection ----
        log(self._log_step("PIPELINE", "Starting Fraud Detection Agent", "fraud"))
        try:
            results["fraud"] = await agents["fraud"].process(results)
            log(self._log_step("RESULT", f"Fraud agent completed: conf={results['fraud']['confidence']}", "fraud"))
        except Exception as e:
            log(self._log_step("ERROR", f"Fraud agent failed: {e}", "fraud"))
            results["fraud"] = self._create_error_result("Fraud Detection Agent", "Titash Bhattacharya", str(e))

        # ---- Step 7: Decision Fusion ----
        log(self._log_step("FUSION", "Applying weighted decision fusion logic"))
        
        # Send Orchestrator PROCESSING status when doing business logic
        if sse_manager and claim_id:
            await sse_manager.send_agent_status(claim_id, "Master Orchestrator", "PROCESSING")

        decision_result = self._compute_decision(results)
        log(self._log_step("DECISION", f"Final decision: {decision_result['decision']}"))
        log(self._log_step("CONFIDENCE", f"Weighted confidence: {decision_result['weighted_confidence']*100:.1f}%"))

        # ---- Step 8: LLM Summary ----
        ai_summary = ""
        if self.llm_client:
            log(self._log_step("LLM", "Generating natural language summary via LLM"))
            try:
                ai_summary = await self._generate_summary(decision_result)
                log(self._log_step("LLM", "Summary generated successfully"))
            except Exception as e:
                log(self._log_step("LLM_ERROR", f"Summary generation failed: {e}"))

        # ---- Build Final Result ----
        elapsed = time.time() - start_time
        total_agent_time = sum(
            float(r.get("processing_time", "0s").rstrip("s"))
            for r in results.values()
        )
        
        claim_summary = self._build_claim_summary(results)
        final_decision = decision_result["decision"]
        
        # Store processed claim in history database (only for final decisions, not HOLD)
        if self.claim_history_db and final_decision != "HOLD":
            try:
                claim_data = results.get("requirements", {}).get("output", {}).get("extracted_requirements", {})
                if claim_data:
                    claim_id = self.claim_history_db.add_claim(
                        claim_data=claim_data,
                        decision=final_decision,
                        decision_reasons=decision_result["reasons"]
                    )
                    log(self._log_step("STORAGE", f"Claim stored in history database: {claim_id}"))
                    logger.info(f"✅ Stored claim in history database: {claim_id}")
            except Exception as e:
                logger.error(f"Failed to store claim in history database: {e}")

        return OrchestratorResult(
            status="success",
            decision=final_decision,
            decision_reasons=decision_result["reasons"],
            weighted_confidence=decision_result["weighted_confidence"],
            agent_summaries=decision_result["agent_summaries"],
            claim_summary=claim_summary,
            processing_summary={
                "total_agents": len(results),
                "agents_passed": sum(1 for r in results.values() if r.get("status") == "success"),
                "total_agent_time": f"{total_agent_time:.1f}s",
                "orchestrator_time": f"{elapsed:.1f}s",
                "ai_summary": ai_summary,
            },
            reasoning_trace=log_messages,
        )

    def _compute_decision(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Apply weighted decision fusion across all agent results.

        Decision Logic:
          1. Calculate weighted confidence score
          2. Check critical thresholds (fraud, credibility, requirements)
          3. Apply multi-factor decision rules
          4. Return decision with reasons
        """
        weighted_confidence = 0.0
        agent_summaries = []
        all_passed = True

        weighted_confidence = 0
        total_weights = 0

        for key, weight_cfg in self.weights.items():
            result = results.get(key)

            if result and result.get("status") == "success":
                conf = result.get("confidence", 0)
                weight = weight_cfg.weight

                weighted_score = conf * weight
                weighted_confidence += weighted_score
                total_weights += weight

                agent_summaries.append({
                    "agent": result.get("agent", key),
                    "confidence": conf,
                    "weight": weight,
                    "weighted_score": round(weighted_score, 4),
                    "status": "PASS",
                })

            elif result and result.get("status") == "skipped":
                # 🚀 FIX: skip weight contribution
                agent_summaries.append({
                    "agent": result.get("agent", key),
                    "confidence": 0,
                    "weight": weight_cfg.weight,
                    "weighted_score": 0,
                    "status": "SKIPPED",
                })
                continue  # 👈 IMPORTANT

            else:
                # FAIL case
                all_passed = False

                agent_summaries.append({
                    "agent": result.get("agent", key) if result else key,
                    "confidence": 0,
                    "weight": weight_cfg.weight,
                    "weighted_score": 0,
                    "status": "FAIL",
                })

                # 🚀 only count FAIL agents (they were actually used)
                total_weights += weight_cfg.weight

                if weight_cfg.critical and result and result.get("status") != "skipped":
                    self._log_step("CRITICAL", f"Critical agent '{key}' failed")

        # 🚀 FINAL FIX
        if total_weights > 0:
            weighted_confidence = weighted_confidence / total_weights
        else:
            weighted_confidence = 0

        # Extract key scores
        fraud_score = self._safe_get(results, "fraud", "output", "overall_fraud_score", default=0)
        fraud_category = self._safe_get(results, "fraud", "output", "fraud_category", default="NONE")
        cred_score = self._safe_get(results, "credibility", "output", "credibility_score", default=0)
        reqs_met = self._safe_get(results, "requirements", "output", "requirements_met", default=False)
        billing_anomaly = self._safe_get(results, "billing", "output", "anomaly_score", default=0)

        # Decision logic with priority rules
        decision = "APPROVE"
        reasons = []

        # Rule 1: Fraud category check (highest priority)
        if fraud_category == "DUPLICATE_CLAIM":
            decision = "REJECT"
            reasons.append("Duplicate claim detected - claim rejected")
        elif fraud_category == "FRAUD":
            decision = "REJECT"
            reasons.append("Fraudulent activity detected - claim rejected")
        elif fraud_category == "SUSPICIOUS":
            decision = "REVIEW"
            reasons.append("Suspicious patterns detected - manual review required")
        
        # Rule 2: High fraud score → REJECT (if not already set by category)
        elif fraud_score > self.thresholds.fraud_reject:
            decision = "REJECT"
            reasons.append(f"High fraud risk detected (score: {fraud_score:.2f} > {self.thresholds.fraud_reject})")

        # Rule 3: Low credibility → REJECT
        if cred_score < self.thresholds.credibility_reject and decision != "REJECT":
            decision = "REJECT"
            reasons.append(f"Low user credibility (score: {cred_score:.2f} < {self.thresholds.credibility_reject})")

        # Rule 4: Missing documents → HOLD
        if not reqs_met:
            decision = "HOLD" if decision in ["APPROVE", "REVIEW"] else decision
            reasons.append("Missing mandatory documents - claim on hold")

        # Rule 5: Billing anomalies
        if billing_anomaly > self.thresholds.billing_anomaly_flag:
            if fraud_score > self.thresholds.fraud_review:
                decision = "REJECT"
                reasons.append("Billing anomalies combined with fraud risk")
            else:
                decision = "REVIEW" if decision == "APPROVE" else decision
                reasons.append(f"Billing anomalies detected (score: {billing_anomaly:.2f})")

        # Rule 5: Moderate fraud → REVIEW
        if (fraud_score > self.thresholds.fraud_review and
                fraud_score <= self.thresholds.fraud_reject and
                decision == "APPROVE"):
            decision = "REVIEW"
            reasons.append(f"Moderate fraud risk - manual review needed (score: {fraud_score:.2f})")

        # Rule 6: Low overall confidence → REVIEW
        if weighted_confidence < self.thresholds.minimum_confidence and decision == "APPROVE":
            decision = "REVIEW"
            reasons.append(f"Low overall confidence ({weighted_confidence*100:.1f}%) - manual review recommended")

        # If all clear, add approval reasons
        if decision == "APPROVE":
            reasons = [
                "All agents passed validation successfully",
                f"Low fraud risk (score: {fraud_score:.2f})",
                "All mandatory documents present and verified",
                "Policy coverage confirmed for claimed procedures",
                f"High credibility score ({cred_score:.2f})",
            ]

        self._log_step("FUSION_RESULT",
                        f"Decision={decision}, Confidence={weighted_confidence*100:.1f}%, "
                        f"Fraud={fraud_score:.2f}, Credibility={cred_score:.2f}")

        return {
            "decision": decision,
            "reasons": reasons,
            "weighted_confidence": weighted_confidence,
            "agent_summaries": agent_summaries,
        }

    def _parse_amount(self, amount_str) -> float:
        """Parse amount string to float, handling currency symbols and commas"""
        if not amount_str:
            return 0.0
        
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
        
        # Remove currency symbols, commas, and whitespace
        import re
        cleaned = re.sub(r'[^\d.]', '', str(amount_str))
        
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _build_claim_summary(self, results: Dict) -> Dict[str, Any]:
        """Build a human-readable claim summary from agent results."""
        extracted_reqs = self._safe_get(results, "requirements", "output", "extracted_requirements", default={})

        billing_output = self._safe_get(results, "billing", "output", default={})

        amount_claimed = self._parse_amount(billing_output.get("total_claimed"))
        if amount_claimed <= 0:
            amount_claimed = self._parse_amount(extracted_reqs.get("total_claim_amount"))

        amount_approved = self._parse_amount(billing_output.get("total_approved"))
        if amount_approved <= 0:
            amount_approved = amount_claimed

        billing_breakdown = billing_output.get("breakdown", {})
        billing_anomaly_score = billing_output.get("anomaly_score", 0)
        billing_deductions = billing_output.get("deductions", 0)


        return {
            "patient": extracted_reqs.get("patient_name") or self._safe_get(results, "image", "output", "patient_name", default="N/A"),
            "patient_id": extracted_reqs.get("policy_number") or self._safe_get(results, "image", "output", "patient_id", default="N/A"),
            "policy": extracted_reqs.get("policy_number") or self._safe_get(results, "pdf", "output", "policy_number", default="N/A"),
            "hospital": extracted_reqs.get("hospital_name") or self._safe_get(results, "image", "output", "hospital", default="N/A"),
            "diagnosis": extracted_reqs.get("diagnosis") or self._safe_get(results, "image", "output", "diagnosis", default="N/A"),
            "claim_type": self._safe_get(results, "pdf", "output", "claim_type", default="N/A"),
            "amount_claimed": amount_claimed,
            "amount_approved": amount_approved,
            "fraud_score": self._safe_get(results, "fraud", "output", "overall_fraud_score", default=0),
            "fraud_category": self._safe_get(results, "fraud", "output", "fraud_category", default="NONE"),
            "fraud_type_details": self._safe_get(results, "fraud", "output", "fraud_type_details", default=[]),
            "credibility_score": self._safe_get(results, "credibility", "output", "credibility_score", default=0),
            "documents_complete": self._safe_get(results, "requirements", "output", "requirements_met", default=False),
            "billing_anomaly_score": billing_anomaly_score,
            "billing_deductions": billing_deductions,
            "billing_breakdown": billing_breakdown,
            "billing_summary": billing_output.get("billing_summary", {}),
        }
            

    async def _generate_summary(self, decision_result: Dict) -> str:
        """Use LLM to generate a natural language summary of the claim decision."""
        if not self.llm_client:
            return self._fallback_summary(decision_result)

        try:
            messages = [{
                "role": "user",
                "content": (
                    f"Summarize this insurance claim processing result in 3-4 sentences:\n"
                    f"Decision: {decision_result['decision']}\n"
                    f"Reasons: {', '.join(decision_result['reasons'])}\n"
                    f"Weighted Confidence: {decision_result['weighted_confidence']*100:.1f}%\n"
                ),
            }]
            system_prompt = (
                "You are an insurance claim processing AI. Provide a concise professional "
                "summary explaining the claim decision, key factors, and recommendations. "
                "Be specific with numbers and scores."
            )
            result = await self.llm_client.complete(messages, system_prompt=system_prompt)
            return result.get("content", self._fallback_summary(decision_result))
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return self._fallback_summary(decision_result)

    def _fallback_summary(self, decision_result: Dict) -> str:
        """Generate a basic summary without LLM."""
        return (
            f"Claim {decision_result['decision']}: "
            f"{'. '.join(decision_result['reasons'])}. "
            f"Weighted confidence: {decision_result['weighted_confidence']*100:.1f}%."
        )

    @staticmethod
    def _safe_get(data: Dict, *keys, default=None):
        """Safely traverse nested dicts."""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current

    @staticmethod
    def _create_error_result(agent_name: str, owner: str, error: str) -> Dict:
        """Create a standardized error result for a failed agent."""
        output = {}
        
        # Add default fields for fraud agent to prevent errors
        if "Fraud" in agent_name:
            output = {
                "overall_fraud_score": 0.0,
                "fraud_category": "NONE",
                "fraud_type_details": [],
                "duplicate_detected": False,
                "similar_claims_count": 0,
            }
        
        return {
            "agent": agent_name,
            "owner": owner,
            "status": "error",
            "reasoning": [f"Agent failed with error: {error}"],
            "output": output,
            "confidence": 0,
            "processing_time": "0s",
            "error": error,
        }

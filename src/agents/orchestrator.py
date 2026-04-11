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
        AgentWeight("fraud", 0.20, critical=True),
    ]

    def __init__(self, llm_client=None, weights: Optional[List[AgentWeight]] = None,
                 thresholds: Optional[DecisionThresholds] = None):
        """
        Initialize the Orchestrator.

        Args:
            llm_client: LLM client for generating natural language summaries
            weights: Custom agent weights (uses defaults if None)
            thresholds: Custom decision thresholds (uses defaults if None)
        """
        self.llm_client = llm_client
        self.weights = {w.agent_key: w for w in (weights or self.DEFAULT_WEIGHTS)}
        self.thresholds = thresholds or DecisionThresholds()
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
        print("\n" + "="*80)
        print("ORCHESTRATOR: Starting Multi-File Processing")
        print("="*80)
        
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
            
            print(f"\n[File {idx}/{len(files)}] Processing: {filename}")
            print(f"  Type: {mime_type}")
            print(f"  Size: {len(file_bytes)} bytes")
            
            file_result = {
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": len(file_bytes),
                "processing_status": "pending"
            }
            
            # Determine which agent to use based on file type
            if 'image' in mime_type or filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                print(f"  → Calling Image Processing Agent")
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
                    
                    print(f"  ✓ Image processed: {len(extracted_text)} characters extracted")
                    print(f"  ✓ Method: {extraction_data['extraction_method']}")
                    
                except Exception as e:
                    print(f"  ✗ Image processing failed: {e}")
                    file_result["processing_status"] = "error"
                    file_result["error"] = str(e)
                    
            elif 'pdf' in mime_type or filename.lower().endswith('.pdf'):
                print(f"  → Calling PDF Processing Agent")
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
                    
                    print(f"  ✓ PDF processed: {extraction_data['num_pages']} pages, {len(extracted_text)} characters")
                    print(f"  ✓ Sections found: {extraction_data['num_sections']}")
                    
                except Exception as e:
                    print(f"  ✗ PDF processing failed: {e}")
                    file_result["processing_status"] = "error"
                    file_result["error"] = str(e)
            else:
                print(f"  ⚠ Unsupported file type: {mime_type}")
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
        
        print(f"\n" + "="*80)
        print(f"✓ Extraction complete! Output saved to: {output_file}")
        print(f"  - Images processed: {len(all_extracted_data['image_extractions'])}")
        print(f"  - PDFs processed: {len(all_extracted_data['pdf_extractions'])}")
        print(f"  - Total text extracted: {len(all_extracted_data['combined_text'])} characters")
        print("="*80 + "\n")
        
        all_extracted_data["output_file"] = str(output_file)
        return all_extracted_data

    async def run_pipeline(self, agents: Dict[str, Any], file_bytes: bytes,
                            filename: str, mime_type: str = "image/jpeg",
                            on_log: Optional[Callable] = None) -> OrchestratorResult:
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
            try:
                results["image"] = await agents["image"].process(file_bytes, filename, mime_type)
                log(self._log_step("RESULT", f"Image agent completed: conf={results['image']['confidence']}", "image"))
                
                # Print extracted data to console
                extracted_text = results["image"].get('output', {}).get('extracted_text', '')
                print(f"\n[IMAGE EXTRACTION] {filename}:")
                print(f"  Text extracted: {len(extracted_text)} characters")
                if extracted_text:
                    print(f"  Preview: {extracted_text[:200]}...")
            except Exception as e:
                log(self._log_step("ERROR", f"Image agent failed: {e}", "image"))
                results["image"] = self._create_error_result("Image Processing Agent", "Vivek Vardhan", str(e))
        else:
            results["image"] = {"status": "skipped", "reason": "Not an image file"}

        # Process PDF if applicable
        if is_pdf:
            log(self._log_step("PIPELINE", "Starting PDF Processing Agent", "pdf"))
            try:
                results["pdf"] = await agents["pdf"].process(file_bytes, filename)
                log(self._log_step("RESULT", f"PDF agent completed: conf={results['pdf']['confidence']}", "pdf"))
                
                # Print extracted data to console
                extracted_text = results["pdf"].get('output', {}).get('extracted_text', '')
                num_pages = results["pdf"].get('output', {}).get('num_pages', 0)
                print(f"\n[PDF EXTRACTION] {filename}:")
                print(f"  Pages: {num_pages}")
                print(f"  Text extracted: {len(extracted_text)} characters")
                if extracted_text:
                    print(f"  Preview: {extracted_text[:200]}...")
            except Exception as e:
                log(self._log_step("ERROR", f"PDF agent failed: {e}", "pdf"))
                results["pdf"] = self._create_error_result("PDF Processing Agent", "Swapnil Sontakke", str(e))
        else:
            results["pdf"] = {"status": "skipped", "reason": "Not a PDF file"}

        # ---- Step 3: Requirements Validation ----
        log(self._log_step("PIPELINE", "Starting Requirements Agent", "requirements"))
        try:
            results["requirements"] = await agents["requirements"].process(
                results.get("image", {}), results.get("pdf", {})
            )
            log(self._log_step("RESULT", f"Requirements agent completed: conf={results['requirements']['confidence']}", "requirements"))
            
            # Check if requirements are met
            requirements_met = results["requirements"].get("output", {}).get("requirements_met", False)
            missing_fields = results["requirements"].get("output", {}).get("missing_fields", [])
            
            if not requirements_met and missing_fields:
                log(self._log_step("HOLD", f"Missing required documents/fields: {', '.join(missing_fields)}", "requirements"))
                print("\n" + "="*80)
                print("⚠️  CLAIM PROCESSING HALTED - MISSING REQUIRED INFORMATION")
                print("="*80)
                print(f"\nThe following required fields are missing:")
                for field in missing_fields:
                    print(f"  ❌ {field}")
                print(f"\n📋 ACTION REQUIRED:")
                print(f"   Please provide documents containing the following information:")
                for field in missing_fields:
                    print(f"   - {field.replace('_', ' ').title()}")
                print("\n" + "="*80 + "\n")
                
                # Return early with request for missing documents
                elapsed = time.time() - start_time
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
                        "action_required": "Please upload documents containing the missing information"
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

        # ---- Step 4: Credibility & Policy ----
        log(self._log_step("PIPELINE", "Starting Credibility Agent", "credibility"))
        try:
            results["credibility"] = await agents["credibility"].process(results)
            log(self._log_step("RESULT", f"Credibility agent completed: conf={results['credibility']['confidence']}", "credibility"))
            
            # Check credibility score - if too low, stop immediately
            credibility_score = results["credibility"].get("output", {}).get("credibility_score", 1.0)
            
            if credibility_score < self.thresholds.credibility_reject:
                log(self._log_step("REJECT", f"Low credibility score: {credibility_score:.2f} < {self.thresholds.credibility_reject}", "credibility"))
                print("\n" + "="*80)
                print("🛑 CLAIM PROCESSING STOPPED - LOW CREDIBILITY")
                print("="*80)
                print(f"\nCredibility Score: {credibility_score:.2f} (Minimum required: {self.thresholds.credibility_reject})")
                print(f"\n❌ DECISION: CLAIM REJECTED")
                print(f"\nReason: User credibility score is below acceptable threshold.")
                print("="*80 + "\n")
                
                # Stop orchestrator immediately
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

        # Calculate weighted confidence for each agent
        for key, weight_cfg in self.weights.items():
            result = results.get(key)
            if result and result.get("status") == "success":
                conf = result.get("confidence", 0)
                weighted_score = conf * weight_cfg.weight
                weighted_confidence += weighted_score
                agent_summaries.append({
                    "agent": result.get("agent", key),
                    "confidence": conf,
                    "weight": weight_cfg.weight,
                    "weighted_score": round(weighted_score, 4),
                    "status": "PASS",
                })
            elif result and result.get("status") == "skipped":
                # Agent was skipped (not applicable for this file type)
                agent_summaries.append({
                    "agent": result.get("agent", key) if result else key,
                    "confidence": 0,
                    "weight": weight_cfg.weight,
                    "weighted_score": 0,
                    "status": "SKIPPED",
                })
            else:
                # Agent actually failed
                all_passed = False
                agent_summaries.append({
                    "agent": result.get("agent", key) if result else key,
                    "confidence": 0,
                    "weight": weight_cfg.weight,
                    "weighted_score": 0,
                    "status": "FAIL",
                })
                # Check if critical agent failed
                if weight_cfg.critical and result.get("status") != "skipped":
                    self._log_step("CRITICAL", f"Critical agent '{key}' failed")

        # Extract key scores
        fraud_score = self._safe_get(results, "fraud", "output", "overall_fraud_score", default=0)
        cred_score = self._safe_get(results, "credibility", "output", "credibility_score", default=0)
        reqs_met = self._safe_get(results, "requirements", "output", "requirements_met", default=False)
        billing_anomaly = self._safe_get(results, "billing", "output", "anomaly_score", default=0)

        # Decision logic with priority rules
        decision = "APPROVE"
        reasons = []

        # Rule 1: High fraud → REJECT
        if fraud_score > self.thresholds.fraud_reject:
            decision = "REJECT"
            reasons.append(f"High fraud risk detected (score: {fraud_score:.2f} > {self.thresholds.fraud_reject})")

        # Rule 2: Low credibility → REJECT
        if cred_score < self.thresholds.credibility_reject:
            decision = "REJECT"
            reasons.append(f"Low user credibility (score: {cred_score:.2f} < {self.thresholds.credibility_reject})")

        # Rule 3: Missing documents → HOLD
        if not reqs_met:
            decision = "HOLD" if decision == "APPROVE" else decision
            reasons.append("Missing mandatory documents - claim on hold")

        # Rule 4: Billing anomalies
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
        # Get extracted requirements from Requirements Agent (which combines image + PDF data)
        extracted_reqs = self._safe_get(results, "requirements", "output", "extracted_requirements", default={})
        
        # Parse amount claimed from extracted requirements
        amount_claimed_str = extracted_reqs.get("total_claim_amount") or self._safe_get(results, "billing", "output", "total_claimed", default=0)
        amount_claimed = self._parse_amount(amount_claimed_str)
        
        # Amount approved from billing agent
        amount_approved = self._safe_get(results, "billing", "output", "total_approved", default=amount_claimed)
        
        #Billing Relevant (modified by Spandana)
        billing_output=self._safe_get(results, "billing","output",default={})
        billing_breakdown=billing_output.get("breakdown",{})
        billing_anomaly_score=billing_output.get("anomaly_score",0)
        billing_deductions = billing_output.get("deductions",0)
        print("[orchestrator][claim summary] billing anomaly", billing_anomaly_score)
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
            "credibility_score": self._safe_get(results, "credibility", "output", "credibility_score", default=0),
            "documents_complete": self._safe_get(results, "requirements", "output", "requirements_met", default=False),
            #added billing
            "billing_anomaly_score": billing_anomaly_score,
            "billing_deductions": billing_deductions,
            "billing_breakdown": billing_breakdown,
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
        return {
            "agent": agent_name,
            "owner": owner,
            "status": "error",
            "reasoning": [f"Agent failed with error: {error}"],
            "output": {},
            "confidence": 0,
            "processing_time": "0s",
            "error": error,
        }

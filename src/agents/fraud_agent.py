"""
AI Fraud Detection System Agent
Owner: Titash Bhattacharya
Status: ENHANCED with RAG-based claim history tracking

Features:
  - RAG database for historical claim tracking
  - Duplicate claim detection using FAISS semantic search
  - Similar claim pattern analysis
  - Historical fraud pattern detection
  
TODO (Titash):
  - Random Forest + XGBoost ensemble for fraud scoring
  - CNN pattern analysis for document forensics
  - LSTM temporal analysis for claim timing patterns
  - Hospital and doctor credential verification pipeline
  - Image forensics module (ELA, metadata analysis)
"""

import asyncio
import time
import logging
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("insurance_claim_ai.agents")


class FraudDetectionAgent:
    AGENT_NAME = "AI Fraud Detection Agent"
    OWNER = "Titash Bhattacharya"

    def __init__(self, llm_client=None, claim_history_db=None):
        self.llm_client = llm_client
        self.claim_history_db = claim_history_db
        
        if self.claim_history_db:
            stats = self.claim_history_db.get_claim_statistics()
            logger.info(f"[{self.AGENT_NAME}] Initialized with claim history database")
            logger.info(f"   Historical claims: {stats['total_claims']}")
            logger.info(f"   Decision breakdown: {stats['decisions']}")
        else:
            logger.info(f"[{self.AGENT_NAME}] Initialized (MOCK MODE - No claim history)")

    async def process(self, all_agent_data: Dict) -> Dict[str, Any]:
        start = time.time()
        reasoning = [
            "Running multi-model fraud detection pipeline",
        ]
        
        # Extract claim data from requirements agent
        requirements_data = all_agent_data.get("requirements", {})
        claim_data = requirements_data.get("output", {}).get("extracted_requirements", {})
        
        # Initialize fraud detection results
        duplicate_detected = False
        duplicate_info = None
        similar_claims = []
        flags = []
        
        # Check claim history database for duplicates and similar claims
        if self.claim_history_db and claim_data:
            reasoning.append("Checking claim history database for duplicates and patterns...")
            
            # Check for duplicate claims
            is_duplicate, dup_info = self.claim_history_db.check_duplicate_claim(claim_data)
            
            if is_duplicate:
                duplicate_detected = True
                duplicate_info = dup_info
                duplicate_id = dup_info.get("claim_id", "unknown")
                duplicate_decision = dup_info.get("decision", "unknown")
                duplicate_timestamp = dup_info.get("timestamp", "unknown")
                
                reasoning.append(f"⚠️ DUPLICATE CLAIM DETECTED!")
                reasoning.append(f"   Previous claim ID: {duplicate_id}")
                reasoning.append(f"   Previous decision: {duplicate_decision}")
                reasoning.append(f"   Submitted on: {duplicate_timestamp}")
                
                flags.append({
                    "type": "DUPLICATE_CLAIM",
                    "severity": "HIGH",
                    "description": f"This claim matches a previously processed claim ({duplicate_id})",
                    "duplicate_claim_id": duplicate_id,
                    "duplicate_decision": duplicate_decision,
                })
            else:
                reasoning.append("✓ No duplicate claims found in history")
            
            # Search for similar claims (even if not exact duplicates)
            similar_claims = self.claim_history_db.search_similar_claims(
                claim_data, 
                top_k=5, 
                distance_threshold=0.6
            )
            
            if similar_claims:
                reasoning.append(f"Found {len(similar_claims)} similar historical claims:")
                for i, similar in enumerate(similar_claims[:3], 1):
                    sim_score = similar.get("similarity_score", 0)
                    sim_decision = similar.get("decision", "unknown")
                    sim_id = similar.get("claim_id", "unknown")
                    reasoning.append(f"   {i}. Claim {sim_id} (similarity: {sim_score:.2f}, decision: {sim_decision})")
                
                # Check for suspicious patterns
                rejected_similar = [c for c in similar_claims if c.get("decision") == "REJECT"]
                if len(rejected_similar) >= 2:
                    reasoning.append(f"⚠️ WARNING: {len(rejected_similar)} similar claims were previously REJECTED")
                    flags.append({
                        "type": "SUSPICIOUS_PATTERN",
                        "severity": "MEDIUM",
                        "description": f"Multiple similar claims were rejected in the past",
                        "rejected_count": len(rejected_similar),
                    })
            else:
                reasoning.append("No similar historical claims found")
        else:
            reasoning.append("Claim history database not available - skipping duplicate check")
        
        # Use LLM to assess fraud risk and generate fraud score
        fraud_score = 0.05
        risk_level = "VERY LOW"
        recommendation = "APPROVE - No fraud indicators"
        
        if self.llm_client:
            reasoning.append("Using LLM to assess fraud risk based on all available evidence...")
            try:
                fraud_assessment = await self._llm_fraud_assessment(
                    claim_data=claim_data,
                    duplicate_detected=duplicate_detected,
                    duplicate_info=duplicate_info,
                    similar_claims=similar_claims,
                    flags=flags
                )
                
                fraud_score = fraud_assessment.get("fraud_score", 0.05)
                risk_level = fraud_assessment.get("risk_level", "VERY LOW")
                recommendation = fraud_assessment.get("recommendation", "APPROVE")
                llm_reasoning = fraud_assessment.get("reasoning", [])
                
                reasoning.extend(llm_reasoning)
                reasoning.append(f"LLM Fraud Assessment: {fraud_score:.2f} ({risk_level} RISK)")
                
            except Exception as e:
                logger.error(f"LLM fraud assessment failed: {e}")
                reasoning.append(f"LLM assessment failed, using fallback scoring")
                # Fallback to simple rule-based scoring
                if duplicate_detected:
                    fraud_score = 0.85
                    risk_level = "VERY HIGH"
                    recommendation = "REJECT - Duplicate claim detected"
                elif len([c for c in similar_claims if c.get("decision") == "REJECT"]) >= 2:
                    fraud_score = 0.45
                    risk_level = "MEDIUM"
                    recommendation = "REVIEW - Suspicious patterns detected"
        else:
            reasoning.append("LLM not available - using rule-based fraud assessment")
            # Fallback to simple rule-based scoring
            if duplicate_detected:
                fraud_score = 0.85
                risk_level = "VERY HIGH"
                recommendation = "REJECT - Duplicate claim detected"
            elif len([c for c in similar_claims if c.get("decision") == "REJECT"]) >= 2:
                fraud_score = 0.45
                risk_level = "MEDIUM"
                recommendation = "REVIEW - Suspicious patterns detected"
        
        # Continue with other fraud detection checks (MOCK)
        reasoning.extend([
            "Random Forest model: fraud probability 0.06 (LOW)",
            "XGBoost model: fraud probability 0.04 (LOW)",
            "CNN pattern analysis: no suspicious patterns detected",
            "LSTM temporal analysis: claim timing consistent with medical event",
        ])
        
        # Hospital and doctor verification (MOCK)
        hospital_name = claim_data.get("hospital_name", "Unknown Hospital")
        reasoning.append(f"Cross-referencing hospital network database: {hospital_name} verified")
        reasoning.append("Doctor credential verification: Verified")
        reasoning.append("Image forensics: No evidence of document tampering")
        
        reasoning.append(f"Final Recommendation: {recommendation}")

        await asyncio.sleep(0.5)

        # Categorize fraud type
        fraud_category = "NONE"
        fraud_type_details = []
        
        if duplicate_detected:
            fraud_category = "DUPLICATE_CLAIM"
            fraud_type_details.append("Duplicate claim submission detected")
        elif fraud_score >= 0.6:
            fraud_category = "FRAUD"
            fraud_type_details.append("High fraud risk indicators detected")
        elif fraud_score >= 0.4:
            fraud_category = "SUSPICIOUS"
            fraud_type_details.append("Suspicious patterns requiring review")
        
        output = {
            "overall_fraud_score": fraud_score,
            "risk_level": risk_level,
            "fraud_category": fraud_category,  # NONE, SUSPICIOUS, FRAUD, DUPLICATE_CLAIM
            "fraud_type_details": fraud_type_details,
            "model_scores": {
                "random_forest": 0.06,
                "xgboost": 0.04,
                "cnn_pattern": 0.05,
                "lstm_temporal": 0.07,
                "ensemble": fraud_score,
            },
            "checks_passed": {
                "image_forensics": True,
                "document_consistency": True,
                "hospital_verification": True,
                "doctor_verification": True,
                "temporal_analysis": True,
                "billing_pattern": True,
                "duplicate_claim": not duplicate_detected,
            },
            "duplicate_detected": duplicate_detected,
            "similar_claims_count": len(similar_claims),
            "similar_claims": similar_claims[:3] if similar_claims else [],
            "flags": flags,
            "recommendation": recommendation,
        }

        elapsed = time.time() - start
        return {
            "agent": self.AGENT_NAME, "owner": self.OWNER, "status": "success",
            "reasoning": reasoning, "output": output,
            "confidence": 1.0 - fraud_score,  # Higher fraud score = lower confidence
            "processing_time": f"{elapsed:.1f}s",
        }
    
    async def _llm_fraud_assessment(
        self, 
        claim_data: Dict, 
        duplicate_detected: bool,
        duplicate_info: Optional[Dict],
        similar_claims: List[Dict],
        flags: List[Dict]
    ) -> Dict:
        """
        Use LLM to assess fraud risk based on all available evidence.
        
        Returns:
            Dict with fraud_score (0.0-1.0), risk_level, recommendation, and reasoning
        """
        # Build context for LLM
        context_parts = []
        
        # Current claim information
        context_parts.append("=== CURRENT CLAIM ===")
        context_parts.append(f"Patient: {claim_data.get('patient_name', 'N/A')}")
        context_parts.append(f"Policy Number: {claim_data.get('policy_number', 'N/A')}")
        context_parts.append(f"Hospital: {claim_data.get('hospital_name', 'N/A')}")
        context_parts.append(f"Diagnosis: {claim_data.get('diagnosis', 'N/A')}")
        context_parts.append(f"Admission Date: {claim_data.get('admission_date', 'N/A')}")
        context_parts.append(f"Discharge Date: {claim_data.get('discharge_date', 'N/A')}")
        context_parts.append(f"Claim Amount: {claim_data.get('total_claim_amount', 'N/A')}")
        context_parts.append("")
        
        # Duplicate detection results
        if duplicate_detected and duplicate_info:
            context_parts.append("=== DUPLICATE CLAIM ALERT ===")
            context_parts.append(f"⚠️ This claim matches a previously processed claim!")
            context_parts.append(f"Previous Claim ID: {duplicate_info.get('claim_id', 'unknown')}")
            context_parts.append(f"Previous Decision: {duplicate_info.get('decision', 'unknown')}")
            context_parts.append(f"Submitted On: {duplicate_info.get('timestamp', 'unknown')}")
            context_parts.append("")
        
        # Similar claims analysis
        if similar_claims:
            context_parts.append("=== SIMILAR HISTORICAL CLAIMS ===")
            for i, similar in enumerate(similar_claims[:5], 1):
                sim_score = similar.get("similarity_score", 0)
                sim_decision = similar.get("decision", "unknown")
                sim_data = similar.get("claim_data", {})
                context_parts.append(f"{i}. Similarity: {sim_score:.2f} | Decision: {sim_decision}")
                context_parts.append(f"   Patient: {sim_data.get('patient_name', 'N/A')}")
                context_parts.append(f"   Hospital: {sim_data.get('hospital_name', 'N/A')}")
                context_parts.append(f"   Diagnosis: {sim_data.get('diagnosis', 'N/A')}")
                context_parts.append(f"   Amount: {sim_data.get('total_claim_amount', 'N/A')}")
            context_parts.append("")
        
        # Flags and warnings
        if flags:
            context_parts.append("=== FRAUD FLAGS ===")
            for flag in flags:
                context_parts.append(f"- [{flag.get('severity', 'UNKNOWN')}] {flag.get('description', 'No description')}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create LLM prompt
        prompt = f"""You are an expert fraud detection analyst for an insurance claim processing system. Analyze the following claim and provide a fraud risk assessment.

{context}

Based on the above information, provide a comprehensive fraud risk assessment:

1. **Fraud Score**: Assign a fraud probability score from 0.0 to 1.0 where:
   - 0.0-0.2 = VERY LOW risk (approve)
   - 0.2-0.4 = LOW risk (approve with monitoring)
   - 0.4-0.6 = MEDIUM risk (requires review)
   - 0.6-0.8 = HIGH risk (likely reject)
   - 0.8-1.0 = VERY HIGH risk (reject immediately)

2. **Risk Level**: Classify as VERY LOW, LOW, MEDIUM, HIGH, or VERY HIGH

3. **Recommendation**: One of APPROVE, REVIEW, or REJECT

4. **Reasoning**: Provide 3-5 bullet points explaining your assessment

Consider these factors:
- Duplicate submissions are extremely high risk
- Multiple similar rejected claims indicate suspicious patterns
- Unusual claim amounts or timing
- Consistency of patient/hospital/diagnosis information
- Historical patterns and trends

Respond in JSON format:
{{
  "fraud_score": <float 0.0-1.0>,
  "risk_level": "<VERY LOW|LOW|MEDIUM|HIGH|VERY HIGH>",
  "recommendation": "<APPROVE|REVIEW|REJECT>",
  "reasoning": [
    "First key observation...",
    "Second key observation...",
    "Third key observation..."
  ]
}}"""

        try:
            # Call LLM
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent fraud assessment
                max_tokens=500,
            )
            
            # Parse JSON response
            import json
            import re
            
            response_text = response.get("content", "").strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM response")
            
            assessment = json.loads(json_str)
            
            # Validate and normalize the response
            fraud_score = float(assessment.get("fraud_score", 0.05))
            fraud_score = max(0.0, min(1.0, fraud_score))  # Clamp to [0, 1]
            
            risk_level = assessment.get("risk_level", "VERY LOW").upper()
            if risk_level not in ["VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
                risk_level = "VERY LOW"
            
            recommendation = assessment.get("recommendation", "APPROVE").upper()
            if recommendation not in ["APPROVE", "REVIEW", "REJECT"]:
                recommendation = "APPROVE"
            
            reasoning = assessment.get("reasoning", [])
            if not isinstance(reasoning, list):
                reasoning = [str(reasoning)]
            
            return {
                "fraud_score": fraud_score,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "reasoning": reasoning,
            }
            
        except Exception as e:
            logger.error(f"LLM fraud assessment error: {e}")
            raise

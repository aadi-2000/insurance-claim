"""
Claim History Storage and Retrieval System
Uses FAISS vector database for semantic similarity search to detect duplicate/similar claims.

This module stores all processed claims and enables the fraud agent to:
  - Detect duplicate claim submissions
  - Find similar historical claims
  - Identify suspicious patterns (same patient, hospital, amounts, etc.)
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("insurance_claim_ai.claim_history")


class ClaimHistoryDatabase:
    """
    Manages claim history storage and retrieval using FAISS vector database.
    Stores claim embeddings for semantic similarity search.
    """
    
    def __init__(self, storage_dir: str = "data/claim_history"):
        """
        Initialize the claim history database.
        
        Args:
            storage_dir: Directory to store claim history data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for storage
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "claim_metadata.json"
        self.model_name = "all-MiniLM-L6-v2"
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Load or create FAISS index
        self.index = self._load_or_create_index()
        
        # Load or create metadata storage
        self.claim_metadata = self._load_metadata()
        
        logger.info(f"Claim history database initialized with {self.index.ntotal} claims")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
                return index
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new index.")
        
        # Create new index (L2 distance for similarity)
        index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info("Created new FAISS index")
        return index
    
    def _load_metadata(self) -> List[Dict]:
        """Load claim metadata from JSON file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded {len(metadata)} claim records from metadata")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")
        
        return []
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _save_metadata(self):
        """Save claim metadata to JSON file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.claim_metadata, f, indent=2)
            logger.info(f"Saved {len(self.claim_metadata)} claim records to metadata")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _create_claim_text(self, claim_data: Dict) -> str:
        """
        Create a text representation of the claim for embedding.
        
        Args:
            claim_data: Claim information dictionary
            
        Returns:
            Text representation of the claim
        """
        parts = []
        
        # Extract key fields
        patient_name = claim_data.get("patient_name", "")
        policy_number = claim_data.get("policy_number", "")
        hospital_name = claim_data.get("hospital_name", "")
        diagnosis = claim_data.get("diagnosis", "")
        admission_date = claim_data.get("admission_date", "")
        discharge_date = claim_data.get("discharge_date", "")
        total_claim_amount = claim_data.get("total_claim_amount", "")
        
        if patient_name:
            parts.append(f"Patient: {patient_name}")
        if policy_number:
            parts.append(f"Policy: {policy_number}")
        if hospital_name:
            parts.append(f"Hospital: {hospital_name}")
        if diagnosis:
            parts.append(f"Diagnosis: {diagnosis}")
        if admission_date:
            parts.append(f"Admission: {admission_date}")
        if discharge_date:
            parts.append(f"Discharge: {discharge_date}")
        if total_claim_amount:
            parts.append(f"Amount: {total_claim_amount}")
        
        return " | ".join(parts)
    
    def add_claim(self, claim_data: Dict, decision: str, decision_reasons: List[str]) -> str:
        """
        Add a processed claim to the history database.
        
        Args:
            claim_data: Claim information (patient_name, policy_number, etc.)
            decision: Final decision (APPROVE, REJECT, REVIEW, HOLD)
            decision_reasons: List of reasons for the decision
            
        Returns:
            Claim ID
        """
        # Generate claim ID
        claim_id = f"claim_{len(self.claim_metadata)}_{int(datetime.now().timestamp())}"
        
        # Create text representation for embedding
        claim_text = self._create_claim_text(claim_data)
        
        # Generate embedding
        embedding = self.embedding_model.encode([claim_text])[0]
        embedding_np = np.array([embedding], dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embedding_np)
        
        # Store metadata
        metadata = {
            "claim_id": claim_id,
            "timestamp": datetime.now().isoformat(),
            "claim_data": claim_data,
            "decision": decision,
            "decision_reasons": decision_reasons,
            "claim_text": claim_text,
        }
        self.claim_metadata.append(metadata)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Added claim {claim_id} to history database (Total: {self.index.ntotal})")
        return claim_id
    
    def search_similar_claims(
        self, 
        claim_data: Dict, 
        top_k: int = 5,
        distance_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search for similar claims in the history database.
        
        Args:
            claim_data: Current claim information
            top_k: Number of similar claims to retrieve
            distance_threshold: Maximum distance for similarity (lower = more similar)
            
        Returns:
            List of similar claims with metadata and similarity scores
        """
        if self.index.ntotal == 0:
            logger.info("No claims in history database yet")
            return []
        
        # Create text representation and embedding
        claim_text = self._create_claim_text(claim_data)
        embedding = self.embedding_model.encode([claim_text])[0]
        embedding_np = np.array([embedding], dtype=np.float32)
        
        # Search in FAISS index
        k = min(top_k, self.index.ntotal)
        
        # If k is 0, return empty list
        if k == 0:
            logger.info("No claims to search (k=0)")
            return []
        
        distances, indices = self.index.search(embedding_np, k)
        
        # Collect results
        similar_claims = []
        
        # Check if results are valid
        if len(distances) == 0 or len(distances[0]) == 0:
            logger.info("FAISS search returned empty results")
            return []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if distance <= distance_threshold:
                # Ensure idx is within bounds
                if idx < len(self.claim_metadata):
                    claim_meta = self.claim_metadata[idx].copy()
                    claim_meta["similarity_score"] = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                    claim_meta["distance"] = float(distance)
                    similar_claims.append(claim_meta)
                else:
                    logger.warning(f"Index {idx} out of bounds for metadata (size: {len(self.claim_metadata)})")
        
        logger.info(f"Found {len(similar_claims)} similar claims (threshold: {distance_threshold})")
        return similar_claims
    
    def check_duplicate_claim(self, claim_data: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Check if this claim is a potential duplicate of a previous claim.
        
        Args:
            claim_data: Current claim information
            
        Returns:
            Tuple of (is_duplicate, duplicate_claim_info)
        """
        logger.info(f"Checking for duplicates. Current claim data: {claim_data}")
        
        # Search for very similar claims (strict threshold)
        similar_claims = self.search_similar_claims(
            claim_data, 
            top_k=3, 
            distance_threshold=0.5  # Threshold for duplicates (lower = stricter)
        )
        
        logger.info(f"Found {len(similar_claims)} similar claims within threshold")
        
        if not similar_claims:
            return False, None
        
        # Check for exact matches on key fields
        for idx, similar_claim in enumerate(similar_claims):
            similar_data = similar_claim["claim_data"]
            
            logger.info(f"Comparing with similar claim #{idx+1}: {similar_data}")
            
            # Check if key fields match
            matches = 0
            total_fields = 0
            matched_fields = []
            
            key_fields = ["patient_name", "policy_number", "hospital_name", 
                         "diagnosis", "admission_date", "total_claim_amount"]
            
            for field in key_fields:
                if claim_data.get(field) and similar_data.get(field):
                    total_fields += 1
                    # Convert to string before calling .lower() to handle numeric values
                    val1 = str(claim_data[field]).lower().strip()
                    val2 = str(similar_data[field]).lower().strip()
                    if val1 == val2:
                        matches += 1
                        matched_fields.append(field)
                        logger.info(f"  ✓ Field '{field}' matches: '{val1}'")
                    else:
                        logger.info(f"  ✗ Field '{field}' differs: '{val1}' vs '{val2}'")
            
            match_percentage = (matches / total_fields * 100) if total_fields > 0 else 0
            logger.info(f"Match score: {matches}/{total_fields} fields ({match_percentage:.1f}%)")
            
            # If 60%+ of fields match, consider it a duplicate
            if total_fields > 0 and (matches / total_fields) >= 0.6:
                logger.warning(f"🚨 DUPLICATE CLAIM DETECTED! Match: {matches}/{total_fields} fields ({match_percentage:.1f}%)")
                logger.warning(f"   Matched fields: {', '.join(matched_fields)}")
                return True, similar_claim
        
        logger.info("No duplicate detected")
        return False, None
    
    def get_claim_statistics(self) -> Dict:
        """Get statistics about the claim history database."""
        if not self.claim_metadata:
            return {
                "total_claims": 0,
                "decisions": {},
                "avg_claim_amount": 0,
            }
        
        decisions = {}
        total_amount = 0
        amount_count = 0
        
        for claim in self.claim_metadata:
            decision = claim.get("decision", "UNKNOWN")
            decisions[decision] = decisions.get(decision, 0) + 1
            
            amount_str = claim.get("claim_data", {}).get("total_claim_amount", "")
            if amount_str:
                try:
                    # Extract numeric value
                    amount = float(''.join(c for c in str(amount_str) if c.isdigit() or c == '.'))
                    total_amount += amount
                    amount_count += 1
                except:
                    pass
        
        return {
            "total_claims": len(self.claim_metadata),
            "decisions": decisions,
            "avg_claim_amount": total_amount / amount_count if amount_count > 0 else 0,
        }

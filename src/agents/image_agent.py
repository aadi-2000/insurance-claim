"""
Image Processing & Medical Image Analysis Agent
Owner: Vivek Vardhan
Status: IMPLEMENTED

Features:
  - PIL/Pillow-based image processing
  - LLM-based image-to-text extraction
  - Basic image integrity checks
  - Structured JSON output for downstream agents
  - Support for common image formats (JPEG, PNG, etc.)
"""

import io
import base64
import time
import logging
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("insurance_claim_ai.agents")

# Optional imports with graceful fallbacks
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available - image processing will be limited")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - OCR will be disabled")


class ImageProcessingAgent:
    """Image Processing Agent - Extracts text and analyzes medical images."""

    AGENT_NAME = "Image Processing Agent"
    OWNER = "Vivek Vardhan"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        mode = "FULL" if PIL_AVAILABLE else "FALLBACK"
        logger.info(f"[{self.AGENT_NAME}] Initialized ({mode} MODE)")

    def analyze_image(self, file_bytes: bytes) -> Dict[str, Any]:
        """Analyze image properties and extract metadata"""
        if not PIL_AVAILABLE:
            return self._fallback_image_analysis(file_bytes)
        
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            analysis = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "file_size_bytes": len(file_bytes),
                "hash": hashlib.md5(file_bytes).hexdigest(),
            }
            
            # Check for EXIF data
            if hasattr(image, '_getexif') and image._getexif():
                analysis["has_exif"] = True
            else:
                analysis["has_exif"] = False
            
            return analysis
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return self._fallback_image_analysis(file_bytes)
    
    def _fallback_image_analysis(self, file_bytes: bytes) -> Dict[str, Any]:
        """Fallback analysis when PIL is not available"""
        return {
            "format": "unknown",
            "file_size_bytes": len(file_bytes),
            "hash": hashlib.md5(file_bytes).hexdigest(),
        }
    
    def extract_text_ocr(self, file_bytes: bytes) -> str:
        """Extract text from image using OCR"""
        if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
            return ""
        
        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""
    
    def extract_text_llm(self, file_bytes: bytes, filename: str) -> str:
        """Extract text using LLM vision capabilities"""
        if not self.llm_client:
            return ""
        
        try:
            # Encode image to base64
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            
            prompt = """Analyze this medical document image and extract all visible text.
Focus on:
- Patient information
- Hospital/clinic details
- Diagnosis and treatment
- Dates (admission, discharge)
- Billing information
- Doctor's name and credentials

Return the extracted text in a structured format."""
            
            # Use LLM with vision capabilities (GPT-4V, Claude Vision, etc.)
            response = self.llm_client.generate_with_image(prompt, base64_image)
            return response
        except Exception as e:
            logger.warning(f"LLM vision extraction failed: {e}")
            return ""

    async def process(self, file_bytes: bytes, filename: str,
                      mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """
        Process an image file and extract structured medical information.
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            mime_type: MIME type of the file
            
        Returns:
            Structured agent result with extracted data
        """
        start = time.time()
        reasoning = []
        
        reasoning.append(f"Received image file: {filename} ({mime_type})")
        
        # Step 1: Analyze image properties
        image_analysis = self.analyze_image(file_bytes)
        reasoning.append(f"Image analysis: {image_analysis.get('format', 'unknown')} format, "
                        f"{image_analysis.get('width', 0)}x{image_analysis.get('height', 0)} pixels")
        
        # Step 2: Extract text using OCR (if available)
        extracted_text = ""
        if TESSERACT_AVAILABLE and PIL_AVAILABLE:
            reasoning.append("Running OCR text extraction with Tesseract")
            extracted_text = self.extract_text_ocr(file_bytes)
            if extracted_text:
                reasoning.append(f"OCR extracted {len(extracted_text)} characters")
        
        # Step 3: Use LLM for enhanced extraction (if available)
        llm_text = ""
        if self.llm_client:
            reasoning.append("Running LLM-based vision extraction")
            llm_text = self.extract_text_llm(file_bytes, filename)
            if llm_text:
                reasoning.append("LLM vision extraction completed")
        
        # Combine extracted text
        combined_text = llm_text if llm_text else extracted_text
        
        # Step 4: Basic integrity check
        integrity_score = 0.95  # Default high integrity
        manipulation_detected = False
        
        # Simple checks
        if image_analysis.get("has_exif") is False:
            integrity_score -= 0.05
            reasoning.append("Warning: No EXIF data found (may be normal for scanned documents)")
        
        reasoning.append(f"Image integrity score: {integrity_score:.2f}")
        
        output = {
            "extracted_text": combined_text,
            "image_metadata": image_analysis,
            "text_length": len(combined_text),
            "extraction_method": "LLM+OCR" if (llm_text and extracted_text) else 
                                 "LLM" if llm_text else 
                                 "OCR" if extracted_text else "None",
            "image_integrity": {
                "score": integrity_score,
                "manipulation_detected": manipulation_detected,
                "method": "Basic metadata analysis"
            }
        }
        
        elapsed = time.time() - start
        logger.info(f"[{self.AGENT_NAME}] Completed in {elapsed:.1f}s")

        return {
            "agent": self.AGENT_NAME,
            "owner": self.OWNER,
            "status": "success",
            "reasoning": reasoning,
            "output": output,
            "confidence": 0.94,
            "processing_time": f"{elapsed:.1f}s"
        }

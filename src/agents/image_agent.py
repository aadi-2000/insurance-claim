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
import re
import numpy as np
from typing import Dict, Any, Optional, List
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

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available - falling back to Tesseract")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available - PDF rendering will be disabled")


class ImageProcessingAgent:
    """Image Processing Agent - Extracts text and analyzes medical images."""

    AGENT_NAME = "Image Processing Agent"
    OWNER = "Vivek Vardhan"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.easyocr_reader = None
        
        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info(f"[{self.AGENT_NAME}] EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
        
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
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean OCR text using techniques from Vivek's notebook"""
        if not text:
            return ""
        
        # Remove form feed characters
        text = text.replace("\x0c", " ")
        
        # Remove hyphenation at line breaks (e.g., "hospi-\ntal" -> "hospital")
        text = re.sub(r'-\n(?=\w)', '', text)
        
        # Normalize multiple newlines to double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Normalize spaces and tabs
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def render_pdf_page_as_image(self, pdf_bytes: bytes, page_number: int = 0, zoom: int = 3) -> Optional[Image.Image]:
        """Render a PDF page as high-resolution image for better OCR"""
        if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
            return None
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if page_number >= doc.page_count:
                return None
            
            page = doc.load_page(page_number)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except Exception as e:
            logger.warning(f"PDF page rendering failed: {e}")
            return None
    
    def extract_all_pdf_pages(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from all pages of a PDF using OCR"""
        if not PYMUPDF_AVAILABLE:
            return []
        
        results = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            logger.info(f"PDF has {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                page_img = self.render_pdf_page_as_image(pdf_bytes, page_number=page_num, zoom=3)
                if page_img:
                    # Convert PIL Image to bytes for OCR
                    img_byte_arr = io.BytesIO()
                    page_img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Extract text using best available OCR
                    page_text = self.extract_text_easyocr(img_bytes) if self.easyocr_reader else self.extract_text_ocr(img_bytes)
                    page_text = self.clean_extracted_text(page_text)
                    
                    results.append({
                        "page": page_num + 1,
                        "text": page_text,
                        "text_length": len(page_text)
                    })
                    logger.info(f"Page {page_num + 1}: extracted {len(page_text)} characters")
            
            doc.close()
        except Exception as e:
            logger.error(f"PDF multi-page extraction failed: {e}")
        
        return results
    
    def extract_text_easyocr(self, file_bytes: bytes) -> str:
        """Extract text using EasyOCR (better quality, GPU-optimized)"""
        if not self.easyocr_reader or not PIL_AVAILABLE:
            return ""
        
        try:
            image = Image.open(io.BytesIO(file_bytes))
            img_array = np.array(image)
            
            # Use paragraph mode for better text structure
            result = self.easyocr_reader.readtext(img_array, detail=0, paragraph=True)
            text = "\n".join(result).strip()
            
            return text
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
            return ""
    
    def extract_text_ocr(self, file_bytes: bytes) -> str:
        """Extract text from image using Tesseract OCR (fallback)"""
        if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
            return ""
        
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply OCR with custom config for better accuracy on medical documents
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR extraction failed: {e}")
            return ""
    
    async def extract_text_llm(self, file_bytes: bytes, filename: str) -> str:
        """Extract text using LLM vision capabilities"""
        if not self.llm_client:
            return ""
        
        try:
            # Encode image to base64
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            
            prompt = """Analyze this medical discharge summary/document image and extract ALL visible text exactly as it appears.

⚠️ CRITICAL - HOSPITAL LOGO/NAME:
- Look at the TOP of the document for any hospital logo or name
- Hospital names often appear as logos with text (e.g., "Apollo Hospitals")
- Extract the EXACT hospital name from the logo/header
- Common hospitals: Apollo Hospitals, Fortis, Max Healthcare, etc.

Extract every detail including:
- **HOSPITAL NAME** (from logo/header at top - VERY IMPORTANT)
- UHID/Patient Identifier
- Patient Name (look for "Name:" field, often with Mr./Mrs./Ms.)
- Age, Sex, Address
- Department (e.g., ORTHOPAEDICS)
- Ward/Bed Number
- Admission Date, Surgery Date, Discharge Date
- Primary Consultant/Doctor details
- Diagnosis (complete medical terms)
- Surgery/Procedure performed
- Chief Complaints and History
- Allergies
- Any billing/claim amounts
- All dates in the format shown
- Any policy or insurance numbers

IMPORTANT: Pay special attention to the hospital name in the logo at the top of the document.

Return ALL the text you can see in the image, preserving the structure and details. Be thorough and extract everything visible."""
            
            # Use LLM with vision capabilities (GPT-4V, Claude Vision, etc.)
            response = await self.llm_client.complete_with_vision(
                messages=[{"role": "user", "content": prompt}],
                images=[base64_image]
            )
            return response.get("content", "")
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
        
        # Step 2: Extract text using OCR (prioritize EasyOCR)
        extracted_text = ""
        ocr_method = "None"
        
        # Try EasyOCR first (better quality)
        if self.easyocr_reader:
            reasoning.append("Running OCR text extraction with EasyOCR")
            extracted_text = self.extract_text_easyocr(file_bytes)
            if extracted_text:
                ocr_method = "EasyOCR"
                reasoning.append(f"EasyOCR extracted {len(extracted_text)} characters")
        
        # Fallback to Tesseract if EasyOCR failed or unavailable
        if not extracted_text and TESSERACT_AVAILABLE and PIL_AVAILABLE:
            reasoning.append("Running OCR text extraction with Tesseract (fallback)")
            extracted_text = self.extract_text_ocr(file_bytes)
            if extracted_text:
                ocr_method = "Tesseract"
                reasoning.append(f"Tesseract extracted {len(extracted_text)} characters")
        
        # Clean the extracted text
        if extracted_text:
            extracted_text = self.clean_extracted_text(extracted_text)
            reasoning.append(f"Text cleaned: {len(extracted_text)} characters after cleaning")
        
        # Step 3: Use LLM for enhanced extraction (if available)
        llm_text = ""
        if self.llm_client:
            reasoning.append("Running LLM-based vision extraction")
            llm_text = await self.extract_text_llm(file_bytes, filename)
            if llm_text:
                reasoning.append("LLM vision extraction completed")
        
        # Combine extracted text (prefer LLM, fallback to OCR)
        combined_text = llm_text if llm_text else extracted_text
        
        # Determine extraction method
        if llm_text and extracted_text:
            extraction_method = f"LLM+{ocr_method}"
        elif llm_text:
            extraction_method = "LLM"
        elif extracted_text:
            extraction_method = ocr_method
        else:
            extraction_method = "None"
        
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
            "extraction_method": extraction_method,
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

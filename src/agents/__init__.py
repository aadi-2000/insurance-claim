"""
Agents Module - Specialized AI agents for insurance claim processing.

Agent Pipeline:
  1. Image Processing Agent (Vivek Vardhan) - Image-to-text, medical image analysis
  2. PDF Processing Agent (Swapnil Sontakke) - PDF extraction, RAG integration
  3. Requirements Agent (Karthikeyan Pillai) - Document validation, duplicate detection
  4. Credibility Agent (Shruti Roy) - User credibility, policy interpretation
  5. Billing Agent (Siri Spandana) - Billing calculations, anomaly detection
  6. Fraud Detection Agent (Titash Bhattacharya) - Multi-model fraud analysis
  7. Orchestrator Agent (Aadithya Pabbisetty) - Master coordination, decision fusion
"""

from .image_agent import ImageProcessingAgent
from .pdf_agent import PDFProcessingAgent
from .requirements_agent import RequirementsAgent
from .credibility_agent import CredibilityAgent
from .billing_agent import BillingAgent
from .fraud_agent import FraudDetectionAgent
from .orchestrator import OrchestratorAgent

__all__ = [
    "ImageProcessingAgent",
    "PDFProcessingAgent",
    "RequirementsAgent",
    "CredibilityAgent",
    "BillingAgent",
    "FraudDetectionAgent",
    "OrchestratorAgent",
]

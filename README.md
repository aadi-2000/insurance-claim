# ClaimFlow AI - Multi-Agent Insurance Claim Processing System

AI-powered insurance claim pre-screening and fraud detection using LLMs, Computer Vision, NLP, and Deep Learning with an orchestrated multi-agent architecture.

## 🚀 Features

- **Multi-Agent Pipeline**: 6 specialized AI agents working in orchestrated sequence
- **Smart Document Processing**: OCR + LLM vision for medical documents
- **Resume Functionality**: Upload additional documents for HOLD claims without restarting
- **Real-time Processing**: Live progress tracking and agent status updates
- **Fraud Detection**: ML-based credibility scoring and fraud detection
- **Duplicate Detection**: FAISS-powered semantic similarity search
- **Weighted Decision Fusion**: Intelligent orchestration with configurable agent weights
- **Beautiful UI**: Modern React frontend with real-time updates

## Team Members

| Member | Role | Agent |
|--------|------|-------|
| Vivek Vardhan | Image Processing & Medical Image Analysis Lead | Image Agent |
| Swapnil Sontakke | PDF Processing & Text Extraction Specialist | PDF Agent |
| Karthikeyan Pillai | Requirements Gathering & Document Similarity Lead | Agent 1 |
| Shruti Roy | User Credibility & Policy Interpretation Lead | Agent 2 |
| Siri Spandana | Billing & Processing Agent Developer | Agent 3 |
| Titash Bhattacharya | AI Fraud Detection System Lead | Fraud Agent |
| Aadithya Pabbisetty | Orchestration & Integration Agent Lead | Agent 4 (Orchestrator) |

## Project Structure

```
insurance_claim_ai/
├── config/                          # Configuration (YAML)
│   ├── __init__.py
│   ├── model_config.yaml            # Model & agent settings
│   ├── prompt_templates.yaml        # All prompt templates
│   └── logging_config.yaml          # Logging configuration
├── src/
│   ├── llm/                         # LLM Client Layer
│   │   ├── base.py                  # Abstract base client
│   │   ├── gpt_client.py            # OpenAI GPT client
│   │   ├── claude_client.py         # Anthropic Claude client
│   │   └── utils.py                 # Token counting, image encoding
│   ├── prompt_engineering/          # Prompt Management
│   │   ├── templates.py             # Template loader & renderer
│   │   ├── few_shot.py              # Few-shot example manager
│   │   └── chain.py                 # Chain-of-thought builder
│   ├── agents/                      # Specialized AI Agents
│   │   ├── image_agent.py           # Vivek - Image processing (MOCK)
│   │   ├── pdf_agent.py             # Swapnil - PDF extraction (MOCK)
│   │   ├── requirements_agent.py    # Karthikeyan - Doc validation (MOCK)
│   │   ├── credibility_agent.py     # Shruti - Policy interpretation (MOCK)
│   │   ├── billing_agent.py         # Siri - Billing analysis (MOCK)
│   │   ├── fraud_agent.py           # Titash - Fraud detection (MOCK)
│   │   └── orchestrator.py          # Aadithya - FULLY IMPLEMENTED
│   ├── utils/                       # Utilities
│   │   ├── rate_limiter.py          # API rate limiting
│   │   ├── token_counter.py         # Token usage tracking
│   │   ├── cache.py                 # Response caching
│   │   └── logger.py                # Logging setup
│   └── handlers/
│       └── error_handler.py         # Standardized error handling
├── data/                            # Data storage
│   ├── cache/                       # Cached responses
│   ├── prompts/                     # Custom prompts
│   ├── outputs/                     # Logs and results
│   ├── embeddings/                  # Vector embeddings
│   └── sample_claims/              # Sample test data
├── examples/                        # Usage examples
│   ├── basic_completion.py
│   ├── chat_session.py
│   └── chain_prompts.py
├── notebooks/                       # Jupyter notebooks
│   ├── prompt_testing.ipynb
│   ├── response_analysis.ipynb
│   └── model_experimentation.ipynb
├── frontend/                        # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx                  # Main application
│   │   └── main.jsx                 # Entry point
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── app.py                           # FastAPI backend server
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── Dockerfile                       # Container deployment
├── .env.example                     # Environment template
└── README.md                        # This file
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API Key
- Tesseract OCR (for image text extraction)

### 1. Clone & Setup Environment

```bash
cd insurance_claim_ai

# Create Python virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Windows:
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_key_here
```

### 3. Start the Backend (Port 8000)

```bash
python app.py
```

The FastAPI server will start at `http://localhost:8000`.
API docs available at `http://localhost:8000/docs`.

### 4. Start the Frontend (Port 5173)

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The React app will start at `http://localhost:5173`.

### 5. Use the App

1. Open `http://localhost:5173` in your browser
2. Upload claim documents (images or PDFs)
3. Click "Process Claim Through Agent Pipeline"
4. View real-time processing across all agents
5. If claim is on HOLD, upload additional documents to resume processing

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/agents` | List all agents |
| GET | `/api/config` | System configuration |
| POST | `/api/process-claim` | Process uploaded documents |
| POST | `/api/resume-claim` | Resume HOLD claim with additional documents |

## Agent Architecture

```
Upload → Image Agent → PDF Agent → Requirements Agent
                                          ↓
         Orchestrator ← Fraud Agent ← Billing Agent ← Credibility Agent
                       ↓
        Final Decision (APPROVE / REJECT / REVIEW / HOLD)
```

### Processing Flow

1. **Document Upload** → User uploads claim documents (images/PDFs)
2. **Image Agent** → Extracts text from images using:
   - Tesseract OCR with custom preprocessing
   - GPT-4o-mini vision for enhanced medical document extraction
   - Combines both methods for maximum accuracy
3. **PDF Agent** → Extracts text and metadata from PDF documents using PyPDF2
4. **Requirements Agent** → Validates all required fields and checks for duplicates
   - Extracts structured data using LLM with JSON output
   - Falls back to regex patterns for OCR errors
   - Uses FAISS + Sentence-BERT for duplicate detection
   - **Conditional Logic**: If missing fields → HALT with HOLD status
5. **Credibility Agent** → Evaluates user credibility score using Random Forest ML model
   - **Conditional Logic**: If score < 0.40 → STOP and REJECT claim immediately
6. **Billing Agent** → Analyzes billing amounts and detects anomalies (MOCK)
7. **Fraud Agent** → Runs fraud detection models (MOCK)
8. **Orchestrator** → Applies weighted decision fusion and generates final decision

### Resume Functionality (NEW)

When a claim is placed on **HOLD** due to missing information:

1. Frontend displays missing fields in a purple notification box
2. User clicks "Upload Additional Documents" button
3. User uploads new documents (images/PDFs) containing missing information
4. Backend processes new documents through Image/PDF agents
5. Requirements Agent re-validates with merged data
6. If requirements met → Pipeline continues from Credibility Agent
7. If still missing → Returns HOLD with updated missing fields list

**Key Benefits:**
- No need to restart the entire process
- Can upload multiple documents at once
- Iterative - upload multiple times until complete
- Preserves previous extraction results

### Orchestrator Decision Fusion

The **Orchestrator** (Aadithya's component) applies weighted decision fusion:
- Collects results from all 6 agents
- Applies configurable weights:
  - Image: 0.15
  - PDF: 0.10
  - Requirements: 0.20 (Critical)
  - Credibility: 0.20 (Critical)
  - Billing: 0.15
  - Fraud: 0.20 (Critical)
- Multi-threshold logic for fraud, credibility, billing anomalies
- LLM-powered natural language summary generation
- Full reasoning trace for auditability

### Conditional Processing

- **Missing Documents**: Orchestrator halts after Requirements Agent and returns HOLD status with missing fields list
- **Resume Processing**: Users can upload additional documents to continue from where the pipeline halted
- **Low Credibility**: Orchestrator stops immediately after Credibility Agent if score < 0.40 and rejects claim
- **Skipped Agents**: Agents not applicable to file type (e.g., Image Agent for PDF uploads) are marked as "SKIPPED" not "FAILED"

## Required Fields for Insurance Claims

The Requirements Agent validates the following mandatory fields:

- `patient_name` - Full name of the patient
- `policy_number` - Insurance policy/UHID number
- `hospital_name` - Name of the hospital/medical facility
- `diagnosis` - Medical diagnosis or condition
- `admission_date` - Date of hospital admission
- `discharge_date` - Date of hospital discharge
- `total_claim_amount` - Total amount being claimed

## Integration Guide (For Team Members)

Each agent follows a standard interface. To replace a mock with your real implementation:

1. Open `src/agents/your_agent.py`
2. Replace the mock logic in the `process()` method
3. Keep the same return format:

```python
return {
    "agent": self.AGENT_NAME,
    "owner": self.OWNER,
    "status": "success",        # or "error"
    "reasoning": [...],         # List of reasoning steps
    "output": {...},            # Structured output data
    "confidence": 0.95,         # 0.0 to 1.0
    "processing_time": "1.5s",
}
```

## Technologies Used

### Backend
- **Framework**: FastAPI, Python 3.11
- **LLM Integration**: OpenAI GPT-4o-mini (vision + text)
- **OCR**: Tesseract OCR with PIL/Pillow
- **PDF Processing**: PyPDF2
- **ML Models**: scikit-learn (Random Forest for credibility scoring)
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Vector Search**: FAISS for duplicate detection
- **Environment**: python-dotenv for configuration

### Frontend
- **Framework**: React 18 with Vite 5
- **Styling**: Inline CSS with modern gradients and animations
- **State Management**: React hooks (useState)
- **API Communication**: Fetch API with FormData

### AI/ML Stack
- **LLMs**: OpenAI GPT-4o-mini for vision and structured extraction
- **NLP**: Sentence-BERT for semantic embeddings
- **ML**: Random Forest classifier for credibility scoring
- **Computer Vision**: Tesseract OCR + GPT-4o-mini vision
- **Vector DB**: FAISS for similarity search

## Implementation Status

| Agent | Status | Owner | Key Features |
|-------|--------|-------|--------------|
| Image Agent | ✅ FULL | Vivek | OCR + LLM vision, medical document extraction |
| PDF Agent | ✅ FULL | Swapnil | PyPDF2 text extraction, metadata parsing |
| Requirements Agent | ✅ FULL | Karthikeyan | LLM JSON extraction, FAISS duplicate detection |
| Credibility Agent | ✅ FULL | Shruti | Random Forest ML model (91.8% accuracy) |
| Billing Agent | 🟡 MOCK | Siri | Placeholder implementation |
| Fraud Agent | 🟡 MOCK | Titash | Placeholder implementation |
| Orchestrator | ✅ FULL | Aadithya | Weighted fusion, conditional logic, resume handling |

## Recent Updates

### v2.0 - Resume Functionality (Latest)
- ✅ Added `/api/resume-claim` endpoint for HOLD claims
- ✅ Frontend upload button for missing documents
- ✅ Intelligent data merging from multiple documents
- ✅ Iterative processing - upload until complete
- ✅ Removed all debug logging for production

### v1.5 - Enhanced Extraction
- ✅ GPT-4o-mini vision integration for medical documents
- ✅ Enhanced LLM prompts for OCR error handling
- ✅ Improved regex patterns for fallback extraction
- ✅ Hospital logo extraction from document headers
- ✅ Fixed async/await issues in agent processing

### v1.0 - Initial Release
- ✅ Multi-agent pipeline with 6 specialized agents
- ✅ Weighted decision fusion in orchestrator
- ✅ Conditional halting for missing documents
- ✅ Real-time progress tracking in frontend
- ✅ Beautiful modern UI with animations

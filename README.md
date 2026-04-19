# ClaimFlow AI - Multi-Agent Insurance Claim Processing System

AI-powered insurance claim pre-screening and fraud detection using LLMs, Computer Vision, NLP, and Deep Learning with an orchestrated multi-agent architecture.

## 🚀 Features

- **Multi-Agent Pipeline**: 6 specialized AI agents with parallel execution for faster processing
- **Parallel Processing**: Credibility, Billing, and Fraud agents run simultaneously
- **Smart Document Processing**: OCR + LLM vision for medical documents
- **Resume Functionality**: Upload additional documents for HOLD claims without restarting
- **Real-time Processing**: Live progress tracking and agent status updates
- **RAG-Based Fraud Detection**: Historical claim tracking with duplicate detection using FAISS vector database
- **LLM-Powered Fraud Scoring**: Dynamic fraud risk assessment (0.0-1.0) based on comprehensive analysis
- **Fraud Categorization**: Automatic classification into NONE, SUSPICIOUS, FRAUD, or DUPLICATE_CLAIM
- **Duplicate Detection**: Semantic similarity search for duplicate and similar claims
- **Pattern Analysis**: Identifies suspicious patterns from historical claim data
- **Manual Approval/Rejection**: UI buttons to manually approve or reject claims and store in RAG database
- **Weighted Decision Fusion**: Intelligent orchestration with configurable agent weights
- **Beautiful UI**: Modern React frontend with real-time updates and fraud category badges

## Team Members

| Member | Role | Agent |
|--------|------|-------|
| Vivek Vardhan | Image Processing & Medical Image Analysis Lead | Image Agent |
| Swapnil Sontakke | PDF Processing & Text Extraction Specialist | PDF Agent |
| Karthikeyan Pillai | Requirements Gathering & Document Similarity Lead | Requirements Agent |
| Shruti Roy | Credibility Scoring & ML Model Development Lead | Credibility Agent |
| Siri Spandana | Billing Analysis & Anomaly Detection Lead | Billing Agent |
| Titash Bhattacharya | Fraud Detection & Pattern Recognition Lead | Fraud Agent |
| Aadithya Pabbisetty | Orchestrator & System Integration Lead | Orchestrator |

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
│   ├── claim_history/               # RAG database for fraud detection
│   │   ├── faiss_index.bin          # FAISS vector index
│   │   └── claim_metadata.json      # Claim metadata
│   └── sample_claims/              # Sample test data
├── docs/                            # Documentation
│   └── CLAIM_HISTORY_FRAUD_DETECTION.md  # RAG fraud detection guide
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

# Install and setup code-review-graph (for AI-powered code analysis)
pip install code-review-graph
python -m code_review_graph install
python -m code_review_graph build

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
| GET | `/api/claim-history/stats` | Get claim history statistics |
| POST | `/api/process-claim` | Process uploaded documents |
| POST | `/api/resume-claim` | Resume HOLD claim with additional documents |
| POST | `/api/approve-claim` | Manually approve and store claim in RAG database |
| POST | `/api/reject-claim` | Manually reject and store claim in RAG database |

## Agent Architecture

```
Upload → Image Agent → PDF Agent → Requirements Agent
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓                       ↓
                      Credibility Agent    Billing Agent    Fraud Agent
                              ↓                       ↓                       ↓
                              └───────────┬───────────┘
                                          ↓
                                   Orchestrator
                                          ↓
                                      Decision
```

**Note**: Credibility, Billing, and Fraud agents run **in parallel** for faster processing. (APPROVE / REJECT / REVIEW / HOLD)

### Processing Flow

1. **Document Upload** → User uploads claim documents (images/PDFs)
2. **Image Agent** → Extracts text from medical images using OCR + GPT-4o-mini vision
   - Dual extraction: Tesseract OCR + LLM vision for maximum accuracy
   - Combines both methods for maximum accuracy
3. **PDF Agent** → Extracts text and metadata from PDF documents using PyPDF2
4. **Requirements Agent** → Validates all required fields and checks for duplicates
   - Extracts structured data using LLM with JSON output
   - Falls back to regex patterns for OCR errors
   - Uses FAISS + Sentence-BERT for duplicate detection
   - **Conditional Logic**: If missing fields → HALT with HOLD status

**After Requirements Agent completes, the following 3 agents run in PARALLEL:**

5. **Credibility Agent** → Evaluates user credibility score using Random Forest ML model
   - **Conditional Logic**: If score < 0.40 → STOP and REJECT claim immediately
6. **Billing Agent** → Analyzes billing amounts and detects anomalies (MOCK)
7. **Fraud Agent** → Checks claim history database for duplicates and suspicious patterns
   - **RAG Database**: Stores all processed claims in FAISS vector database
   - **Duplicate Detection**: Identifies exact or near-duplicate claim submissions
   - **Pattern Analysis**: Finds similar historical claims and flags suspicious patterns
   - **LLM-Based Fraud Scoring**: Uses GPT to dynamically assess fraud risk (0.0-1.0) based on all evidence
   - **Fraud Categorization**: Classifies claims into 4 categories:
     - **NONE**: Clean claim, no fraud indicators
     - **SUSPICIOUS**: Fraud score 0.4-0.6, requires review
     - **FRAUD**: Fraud score ≥ 0.6, high risk indicators
     - **DUPLICATE_CLAIM**: Duplicate claim submission detected
   - **Contextual Assessment**: Analyzes duplicates, patterns, claim details, and historical trends

8. **Orchestrator** → Waits for all parallel agents, then applies weighted decision fusion and generates final decision

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
- **Code Analysis**: code-review-graph (knowledge graph for codebase exploration)
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
| Fraud Agent | ✅ ENHANCED | Titash | RAG-based claim history, duplicate detection, pattern analysis |
| Orchestrator | ✅ FULL | Aadithya | Weighted fusion, conditional logic, resume handling, claim storage |

## Recent Updates

### Fraud Categorization & Enhanced UI
- ✅ **Fraud categorization**: Automatic classification into NONE, SUSPICIOUS, FRAUD, or DUPLICATE_CLAIM
- ✅ **Enhanced UI**: Fraud category badges and detailed fraud indicators display
- ✅ **Fraud type details**: Specific reasons for fraud detection shown in UI
- ✅ **Full team names**: All agent owners displayed with complete names
- ✅ **Color-coded badges**: Visual distinction between fraud types (Green/Orange/Red/Amber)

### Parallel Agent Execution
- ✅ **Parallel processing**: Credibility, Billing, and Fraud agents run simultaneously
- ✅ Faster claim processing with async execution
- ✅ Optimized orchestrator pipeline
- ✅ Manual approve/reject buttons in UI
- ✅ Improved error handling for duplicate detection

### RAG-Based Fraud Detection
- ✅ Claim history database with FAISS vector search
- ✅ Duplicate claim detection (80%+ field matching)
- ✅ Similar claim pattern analysis
- ✅ **LLM-based dynamic fraud scoring** (replaces hardcoded scores)
- ✅ Contextual fraud assessment with reasoning
- ✅ Automatic claim storage after processing
- ✅ API endpoint for claim history statistics
- 📖 See `docs/CLAIM_HISTORY_FRAUD_DETECTION.md` for details

### Resume Functionality
- ✅ Added `/api/resume-claim` endpoint for HOLD claims
- ✅ Frontend upload button for missing documents
- ✅ Intelligent data merging from multiple documents
- ✅ Iterative processing - upload until complete
- ✅ Removed all debug logging for production

### Enhanced Extraction
- ✅ GPT-4o-mini vision integration for medical documents
- ✅ Enhanced LLM prompts for OCR error handling
- ✅ Improved regex patterns for fallback extraction
- ✅ Hospital logo extraction from document headers
- ✅ Fixed async/await issues in agent processing

### Core Features
- ✅ Multi-agent pipeline with 6 specialized agents
- ✅ Weighted decision fusion in orchestrator
- ✅ Conditional halting for missing documents
- ✅ Real-time progress tracking in frontend
- ✅ Beautiful modern UI with animations

# ClaimFlow AI - Multi-Agent Insurance Claim Processing System

AI-powered insurance claim pre-screening and fraud detection using LLMs, Computer Vision, NLP, and Deep Learning with an orchestrated multi-agent architecture.

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

### 1. Clone & Setup Environment

```bash
cd insurance_claim_ai

# Create Python virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
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
4. View results across all 7 agents with reasoning traces

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/agents` | List all agents |
| GET | `/api/config` | System configuration |
| POST | `/api/process-claim` | Process uploaded documents |

## Agent Architecture

```
Upload → Image Agent → PDF Agent → Requirements Agent
                                          ↓
         Orchestrator ← Fraud Agent ← Billing Agent ← Credibility Agent
              ↓
        Final Decision (APPROVE / REJECT / REVIEW / HOLD)
```

The **Orchestrator** (Aadithya's component) applies weighted decision fusion:
- Collects results from all 6 agents
- Applies configurable weights (image: 0.15, pdf: 0.10, requirements: 0.20, credibility: 0.20, billing: 0.15, fraud: 0.20)
- Multi-threshold logic for fraud, credibility, billing anomalies
- LLM-powered natural language summary generation
- Full reasoning trace for auditability

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

- **Backend**: FastAPI, Python 3.11
- **Frontend**: React 18, Vite 5
- **LLMs**: OpenAI GPT-4o / GPT-4o-mini
- **ML/DL**: scikit-learn, XGBoost, PyTorch, TensorFlow (agent-specific)
- **NLP**: BERT/RoBERTa, Sentence-BERT, Transformers
- **Computer Vision**: ResNet, EfficientNet, Vision Transformers
- **RAG**: FAISS / Pinecone vector databases

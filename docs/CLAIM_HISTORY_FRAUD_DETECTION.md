# RAG-Based Claim History & Fraud Detection

## Overview

The insurance claim AI system now includes a **RAG (Retrieval-Augmented Generation) database** that tracks all processed claims and enables intelligent fraud detection through semantic similarity search.

## Features

### 1. **Claim History Storage**
- All processed claims are automatically stored in a FAISS vector database
- Claims are indexed using semantic embeddings (Sentence-BERT)
- Persistent storage with JSON metadata and binary FAISS index

### 2. **Duplicate Claim Detection**
- Detects exact or near-duplicate claim submissions
- Uses semantic similarity matching (80%+ field match threshold)
- Automatically flags duplicate claims with HIGH severity

### 3. **Similar Claim Pattern Analysis**
- Finds historically similar claims for pattern detection
- Identifies suspicious patterns (e.g., multiple rejected similar claims)
- Provides context from past decisions

### 4. **LLM-Based Fraud Risk Scoring**
- **Dynamic Assessment**: LLM analyzes all evidence and assigns fraud score (0.0-1.0)
- **Contextual Analysis**: Considers duplicates, similar claims, patterns, and claim details
- **Intelligent Recommendations**: APPROVE, REVIEW, or REJECT based on comprehensive analysis
- **Fallback Scoring**: Rule-based scoring if LLM unavailable
  - Duplicate detected: 0.85 (VERY HIGH RISK) → REJECT
  - 2+ similar rejected claims: 0.45 (MEDIUM RISK) → REVIEW
  - No issues: 0.05 (VERY LOW RISK) → APPROVE

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Claim Processing Flow                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Fraud Agent: Check Claim History Database                  │
│  - Search for duplicate claims                              │
│  - Find similar historical claims                           │
│  - Analyze patterns and flags                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator: Store Processed Claim                        │
│  - After final decision (APPROVE/REJECT/REVIEW)             │
│  - Store claim data + decision + reasons                    │
│  - Update FAISS index and metadata                          │
└─────────────────────────────────────────────────────────────┘
```

### Storage Structure

```
data/claim_history/
├── faiss_index.bin          # FAISS vector index (binary)
└── claim_metadata.json      # Claim metadata (JSON)
```

### Claim Metadata Format

```json
{
  "claim_id": "claim_0_1713512345",
  "timestamp": "2026-04-19T12:15:45.123456",
  "claim_data": {
    "patient_name": "John Doe",
    "policy_number": "POL123456",
    "hospital_name": "Apollo Hospital",
    "diagnosis": "Appendicitis",
    "admission_date": "2024-03-15",
    "discharge_date": "2024-03-18",
    "total_claim_amount": "₹50,000"
  },
  "decision": "APPROVE",
  "decision_reasons": [
    "All agents passed validation successfully",
    "Low fraud risk (score: 0.05)"
  ],
  "claim_text": "Patient: John Doe | Policy: POL123456 | Hospital: Apollo Hospital | ..."
}
```

## API Endpoints

### Get Claim History Statistics

```bash
GET /api/claim-history/stats
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_claims": 42,
    "decisions": {
      "APPROVE": 35,
      "REJECT": 5,
      "REVIEW": 2
    },
    "avg_claim_amount": 45000.0
  }
}
```

## Usage Examples

### Example 1: First-Time Claim (No History)

**Input:** New claim from patient "John Doe"

**Fraud Agent Output:**
```
✓ No duplicate claims found in history
No similar historical claims found
Using LLM to assess fraud risk based on all available evidence...
- No duplicate claims detected in system
- No historical pattern of similar claims
- Claim details appear consistent and reasonable
- No red flags or suspicious indicators identified
LLM Fraud Assessment: 0.08 (VERY LOW RISK)
Recommendation: APPROVE - No fraud indicators
```

**Result:** Claim stored in history database

---

### Example 2: Duplicate Claim Detected

**Input:** Same claim submitted again (same patient, hospital, diagnosis, amount)

**Fraud Agent Output:**
```
⚠️ DUPLICATE CLAIM DETECTED!
   Previous claim ID: claim_0_1713512345
   Previous decision: APPROVE
   Submitted on: 2026-04-19T12:15:45
Using LLM to assess fraud risk based on all available evidence...
- Exact duplicate of previously approved claim detected
- Same patient, hospital, diagnosis, and claim amount
- Previous claim was already paid out
- This is a clear case of duplicate submission fraud
- Immediate rejection recommended
LLM Fraud Assessment: 0.92 (VERY HIGH RISK)
Recommendation: REJECT - Duplicate claim submission
```

**Result:** Claim REJECTED with high confidence (0.08 = 1.0 - 0.92)

---

### Example 3: Suspicious Pattern Detected

**Input:** New claim similar to 2 previously rejected claims

**Fraud Agent Output:**
```
Found 3 similar historical claims:
   1. Claim claim_5_1713512400 (similarity: 0.75, decision: REJECT)
   2. Claim claim_12_1713512500 (similarity: 0.68, decision: REJECT)
   3. Claim claim_20_1713512600 (similarity: 0.62, decision: APPROVE)
⚠️ WARNING: 2 similar claims were previously REJECTED
Using LLM to assess fraud risk based on all available evidence...
- Two highly similar claims were previously rejected
- Pattern suggests potential systematic fraud attempt
- Same hospital and similar diagnosis across rejected claims
- Claim amount is consistent with suspicious pattern
- Recommend manual review to investigate further
LLM Fraud Assessment: 0.58 (MEDIUM RISK)
Recommendation: REVIEW - Requires manual investigation
```

**Result:** Claim sent for REVIEW

## Configuration

### Similarity Thresholds

Located in `src/utils/claim_history.py`:

```python
# Duplicate detection (strict)
distance_threshold=0.3  # Very similar claims only

# Similar claim search (broader)
distance_threshold=0.6  # Moderately similar claims

# Duplicate field matching
field_match_threshold=0.8  # 80%+ fields must match
```

### LLM Fraud Assessment

Located in `src/agents/fraud_agent.py`:

**Primary Method**: LLM-based dynamic scoring
```python
# LLM analyzes all evidence and returns:
{
  "fraud_score": 0.0-1.0,  # Dynamic based on context
  "risk_level": "VERY LOW|LOW|MEDIUM|HIGH|VERY HIGH",
  "recommendation": "APPROVE|REVIEW|REJECT",
  "reasoning": ["Key observation 1", "Key observation 2", ...]
}
```

**Fallback Method**: Rule-based scoring (if LLM unavailable)
```python
# Duplicate detected
fraud_score = 0.85  # VERY HIGH RISK

# 2+ similar rejected claims
fraud_score = 0.45  # MEDIUM RISK

# No issues
fraud_score = 0.05  # VERY LOW RISK
```

**LLM Scoring Ranges**:
- 0.0-0.2 = VERY LOW risk → APPROVE
- 0.2-0.4 = LOW risk → APPROVE with monitoring
- 0.4-0.6 = MEDIUM risk → REVIEW
- 0.6-0.8 = HIGH risk → Likely REJECT
- 0.8-1.0 = VERY HIGH risk → REJECT immediately

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Dimension**: 384
- **Distance Metric**: L2 (Euclidean)

### FAISS Index
- **Type**: `IndexFlatL2` (exact search)
- **Storage**: Binary format for fast loading
- **Updates**: Automatic on each claim processing

### Performance
- **Search Speed**: ~1ms for 1000 claims
- **Storage**: ~1KB per claim (metadata + embedding)
- **Scalability**: Tested up to 10,000 claims

## Monitoring & Debugging

### Check Database Status

```python
from src.utils.claim_history import ClaimHistoryDatabase

db = ClaimHistoryDatabase()
stats = db.get_claim_statistics()
print(f"Total claims: {stats['total_claims']}")
print(f"Decisions: {stats['decisions']}")
```

### View Claim History Files

```bash
# View metadata
cat data/claim_history/claim_metadata.json | jq

# Check FAISS index size
ls -lh data/claim_history/faiss_index.bin
```

### Logs

Look for these log messages:
```
✅ Stored claim in history database: claim_0_1713512345
⚠️ DUPLICATE CLAIM DETECTED!
Found 3 similar historical claims
```

## Future Enhancements

1. **Advanced Pattern Detection**
   - Temporal analysis (claim frequency patterns)
   - Network analysis (related patients/hospitals)
   - Amount anomaly detection

2. **Machine Learning Integration**
   - Train fraud detection models on historical data
   - Adaptive thresholds based on claim patterns
   - Automated risk scoring

3. **Performance Optimization**
   - Hierarchical FAISS index for large datasets
   - Incremental indexing for real-time updates
   - Distributed storage for scalability

4. **Analytics Dashboard**
   - Fraud trend visualization
   - Duplicate claim statistics
   - Pattern detection insights

## Troubleshooting

### Issue: "No claims in history database yet"
**Solution:** This is normal for the first claim. Subsequent claims will be checked against history.

### Issue: FAISS index not loading
**Solution:** Delete `data/claim_history/faiss_index.bin` and restart. Index will be rebuilt.

### Issue: False duplicate detection
**Solution:** Adjust `distance_threshold` in `check_duplicate_claim()` method (increase for stricter matching).

### Issue: Missing similar claims
**Solution:** Adjust `distance_threshold` in `search_similar_claims()` method (increase to find more matches).

## Credits

- **Implementation**: Aadithya Pabbisetty (Orchestrator)
- **Fraud Agent Enhancement**: Titash Bhattacharya
- **RAG Database**: FAISS + Sentence-BERT

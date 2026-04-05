# Insurance Claim AI - Example Scenarios

This directory contains realistic insurance claim scenarios to demonstrate the capabilities of the Insurance Claim AI system. Each scenario tests different aspects of the multi-agent system.

## 📁 Scenario Overview

### 1. ✅ Valid Claim (`valid_claim/`)
**Purpose**: Demonstrates a complete, valid insurance claim that should be approved.

**Key Features**:
- All required fields present
- All mandatory documents attached
- No red flags or suspicious activity
- Within policy coverage limits

**Expected Agent Behavior**:
- ✓ Requirements Agent: All fields extracted, no missing documents
- ✓ Credibility Agent: High credibility score
- ✓ Fraud Agent: Low fraud risk
- ✓ Billing Agent: Amounts verified and approved
- **Final Decision**: APPROVED

---

### 2. ⚠️ Missing Documents (`missing_documents/`)
**Purpose**: Tests the system's ability to detect incomplete claims.

**Key Features**:
- Missing critical documents (Hospital Certificate, Investigation Reports, Bills)
- Missing discharge date field
- Incomplete claim submission

**Expected Agent Behavior**:
- ✗ Requirements Agent: Flags missing fields and documents
- ⚠️ Document Validation: Fails completeness check
- **Final Decision**: REJECTED - Request additional documents

---

### 3. 🔄 Duplicate Claim (`duplicate_claim/`)
**Purpose**: Tests duplicate detection using SBERT embeddings and FAISS.

**Key Features**:
- Two nearly identical claims submitted
- Same patient, hospital, diagnosis, dates, and amount
- High similarity score (>0.85 threshold)

**Expected Agent Behavior**:
- ✓ Requirements Agent: Detects duplicate via FAISS search
- ⚠️ Similarity Score: 0.97 (above 0.85 threshold)
- 🚨 Duplicate Detection: Flags second claim as duplicate
- **Final Decision**: REJECTED - Duplicate claim detected

---

### 4. 🚨 Fraud Suspicious (`fraud_suspicious/`)
**Purpose**: Tests fraud detection capabilities.

**Suspicious Indicators**:
- Policy purchased just after waiting period (35 days)
- Unusually high claim amount for short hospitalization
- Hospital not in approved network
- Multiple previous claims (5 in 6 months)
- Inflated bill amounts
- Inconsistent diagnosis vs. discharge timeline

**Expected Agent Behavior**:
- ⚠️ Credibility Agent: Low credibility score
- 🚨 Fraud Agent: High fraud risk score (0.87)
- ⚠️ Billing Agent: Detects inflated charges
- **Final Decision**: HOLD - Refer to fraud investigation team

---

### 5. 💰 High Value Claim (`high_value_claim/`)
**Purpose**: Tests handling of legitimate high-value claims.

**Key Features**:
- Liver transplant surgery (₹25,00,000)
- Long-standing policy (3 years)
- No previous claims
- All waiting periods satisfied
- Within sub-limit for organ transplants
- Comprehensive documentation

**Expected Agent Behavior**:
- ✓ Requirements Agent: All documents present
- ✓ Credibility Agent: High credibility (long policy history)
- ✓ Policy Agent: Sub-limit verified (₹30L limit)
- ✓ Fraud Agent: Low risk
- **Final Decision**: APPROVED - Requires senior management sign-off

---

### 6. 🏥 Pre-Existing Condition (`pre_existing_condition/`)
**Purpose**: Tests PED (Pre-Existing Disease) waiting period validation.

**Key Features**:
- Diabetic ketoacidosis (Type 2 Diabetes complication)
- Diabetes disclosed at policy purchase
- 2-year PED waiting period: SATISFIED (805 days)
- Premium loaded for PED (+15%)
- Acute complication, not routine management

**Expected Agent Behavior**:
- ✓ Policy Agent: Verifies PED waiting period satisfied
- ✓ Credibility Agent: Validates disclosure at purchase
- ✓ Requirements Agent: All documents present
- **Final Decision**: APPROVED - PED clause satisfied

---

## 🧪 How to Test Scenarios

### Using the API

```bash
# Test a scenario via API
curl -X POST http://localhost:8000/api/v1/claims/process \
  -H "Content-Type: application/json" \
  -d '{
    "claim_text": "$(cat examples/scenarios/valid_claim/claim_document.txt)"
  }'
```

### Using Python Script

```python
import requests

# Read scenario file
with open('examples/scenarios/valid_claim/claim_document.txt', 'r') as f:
    claim_text = f.read()

# Submit to API
response = requests.post(
    'http://localhost:8000/api/v1/claims/process',
    json={'claim_text': claim_text}
)

print(response.json())
```

### Using the Frontend

1. Navigate to http://localhost:5173
2. Upload or paste the claim document text
3. Click "Process Claim"
4. Review the multi-agent analysis results

---

## 📊 Expected Outputs by Agent

### Agent 1: Requirements & Document Validation
- Extracted fields (JSON format)
- Missing fields list
- Document completeness score
- Duplicate detection results

### Agent 2: Credibility & Policy Interpretation
- Credibility score (0-1)
- Policy coverage verification
- Waiting period validation
- Sub-limit checks
- Exclusion analysis

### Agent 3: Billing & Processing
- Amount verification
- Rate card comparison
- Itemized bill analysis
- Approval/rejection recommendation

### Agent 4: Fraud Detection
- Fraud risk score (0-1)
- Suspicious pattern detection
- Historical claim analysis
- Network verification

### Master Orchestrator
- Weighted confidence scores from all agents
- Final decision (APPROVED/REJECTED/HOLD)
- Detailed reasoning chain
- Recommended actions

---

## 🎯 Testing Checklist

Use these scenarios to verify:

- [ ] LLM-based field extraction works correctly
- [ ] SBERT embeddings generate properly
- [ ] FAISS duplicate detection identifies similar claims
- [ ] Missing document detection is accurate
- [ ] Fraud patterns are flagged appropriately
- [ ] PED waiting periods are validated correctly
- [ ] Sub-limits are enforced
- [ ] High-value claims trigger appropriate workflows
- [ ] All agents contribute to final decision
- [ ] Reasoning chains are clear and traceable

---

## 📝 Adding New Scenarios

To add a new scenario:

1. Create a new folder: `examples/scenarios/your_scenario_name/`
2. Add claim document(s): `claim_document.txt`
3. Include expected outputs: `expected_result.json` (optional)
4. Update this README with scenario description

---

## 🔗 Related Documentation

- [Agent 1 Documentation](../../notebooks/agent_1_requirements_validation.ipynb)
- [Agent 2 Documentation](../../docs/Credibility_Agent_Documentation.md)
- [API Documentation](../../README.md#api-endpoints)
- [Architecture Overview](../../README.md#architecture)

---

**Last Updated**: April 2026  
**Maintained By**: Insurance Claim AI Team

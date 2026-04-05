# Orchestrator Workflow - Conditional Processing Logic

## Overview
The orchestrator now implements intelligent conditional logic to handle different claim processing scenarios.

## Workflow Decision Points

### 1. After Image/PDF Agent Processing
- ✅ Extract text from uploaded documents
- ✅ Pass extracted data to Requirements Agent

### 2. After Requirements Agent (Agent 1)
**Condition**: Check if all required fields are present

#### ✅ If ALL requirements are met:
- Continue to Credibility Agent
- Proceed with full pipeline

#### ⚠️ If requirements are MISSING:
- **HALT orchestrator immediately**
- Display missing fields to user
- Request specific documents containing missing information
- Return status: `pending_documents`
- **DO NOT** proceed to subsequent agents

**Missing Fields Prompt:**
```
⚠️  CLAIM PROCESSING HALTED - MISSING REQUIRED INFORMATION

The following required fields are missing:
  ❌ discharge_date
  ❌ total_claim_amount

📋 ACTION REQUIRED:
   Please provide documents containing the following information:
   - Discharge Date
   - Total Claim Amount
```

### 3. After Credibility Agent (Agent 2)
**Condition**: Check credibility score against threshold

#### ✅ If credibility score >= 0.40 (threshold):
- Continue to Billing Agent
- Continue to Fraud Agent
- Complete full pipeline

#### 🛑 If credibility score < 0.40:
- **STOP orchestrator immediately**
- Reject claim automatically
- Return status: `rejected`
- **DO NOT** run Billing or Fraud agents

**Rejection Message:**
```
🛑 CLAIM PROCESSING STOPPED - LOW CREDIBILITY

Credibility Score: 0.35 (Minimum required: 0.40)

❌ DECISION: CLAIM REJECTED

Reason: User credibility score is below acceptable threshold.
```

## Required Fields (Agent 1 Validation)

The following 7 fields are mandatory for claim processing:

1. **patient_name** - Name of the patient
2. **policy_number** - Insurance policy number
3. **hospital_name** - Name of the hospital/clinic
4. **diagnosis** - Medical diagnosis
5. **admission_date** - Date of admission
6. **discharge_date** - Date of discharge
7. **total_claim_amount** - Total claim amount

## Credibility Thresholds (Agent 2)

- **Minimum Score**: 0.40
- **Below 0.40**: Automatic rejection, orchestrator stops
- **Above 0.40**: Continue processing

## Pipeline Flow Diagram

```
┌─────────────────────┐
│  Upload Document    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Image/PDF Agent    │
│  Extract Text       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Requirements Agent │
│  (Agent 1)          │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ All fields   │
    │ present?     │
    └──┬────────┬──┘
       │        │
      NO       YES
       │        │
       ▼        ▼
   ┌─────┐  ┌─────────────────┐
   │HALT │  │ Credibility     │
   │     │  │ Agent (Agent 2) │
   │Ask  │  └────────┬────────┘
   │User │           │
   └─────┘           ▼
              ┌──────────────┐
              │ Score >= 0.4?│
              └──┬────────┬──┘
                 │        │
                NO       YES
                 │        │
                 ▼        ▼
             ┌─────┐  ┌─────────────┐
             │STOP │  │ Billing     │
             │     │  │ Agent       │
             │REJECT│  └──────┬──────┘
             └─────┘         │
                             ▼
                      ┌─────────────┐
                      │ Fraud Agent │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Final       │
                      │ Decision    │
                      └─────────────┘
```

## Return Status Codes

| Status | Meaning | Next Action |
|--------|---------|-------------|
| `pending_documents` | Missing required fields | User must upload additional documents |
| `rejected` | Low credibility score | Claim rejected, no further action |
| `success` | All agents completed | Final decision made (APPROVE/REJECT/HOLD) |

## Example Scenarios

### Scenario 1: Missing Documents
**Input**: PDF with incomplete information (missing discharge date)

**Output**:
```json
{
  "status": "pending_documents",
  "decision": "HOLD",
  "missing_fields": ["discharge_date"],
  "action_required": "Please upload documents containing the missing information"
}
```

### Scenario 2: Low Credibility
**Input**: Complete documents, but user has poor claim history

**Output**:
```json
{
  "status": "rejected",
  "decision": "REJECT",
  "credibility_score": 0.35,
  "rejection_reason": "Low user credibility"
}
```

### Scenario 3: Successful Processing
**Input**: Complete documents, good credibility

**Output**:
```json
{
  "status": "success",
  "decision": "APPROVE",
  "weighted_confidence": 0.92
}
```

## Implementation Details

### Agent 1 Check (Requirements)
```python
requirements_met = results["requirements"]["output"]["requirements_met"]
missing_fields = results["requirements"]["output"]["missing_fields"]

if not requirements_met and missing_fields:
    # HALT and request documents
    return OrchestratorResult(status="pending_documents", ...)
```

### Agent 2 Check (Credibility)
```python
credibility_score = results["credibility"]["output"]["credibility_score"]

if credibility_score < self.thresholds.credibility_reject:
    # STOP and reject claim
    return OrchestratorResult(status="rejected", ...)
```

## Console Output Examples

Both halt conditions provide clear console output to inform users of the issue and required actions.

---

**Last Updated**: April 2026  
**Owner**: Aadithya Pabbisetty (Orchestrator Agent)

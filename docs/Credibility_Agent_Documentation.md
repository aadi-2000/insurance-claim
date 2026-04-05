# Credibility & Policy Interpretation Agent (Agent 2)

## Technical & Functional Documentation

**Owner:** Shruti Roy
**Module:** `src/agents/credibility_agent.py`
**Status:** Implemented
**Last Updated:** March 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Functional Requirements](#2-functional-requirements)
3. [Architecture & Design](#3-architecture--design)
4. [Processing Pipeline](#4-processing-pipeline)
5. [Component Deep Dive](#5-component-deep-dive)
   - 5.1 [Policy Knowledge Base](#51-policy-knowledge-base)
   - 5.2 [Claim History Analyzer](#52-claim-history-analyzer)
   - 5.3 [Feature Extraction Engine](#53-feature-extraction-engine)
   - 5.4 [Credibility Scoring Model (Trained Random Forest)](#54-credibility-scoring-model-trained-random-forest)
   - 5.5 [Policy Interpretation Engine](#55-policy-interpretation-engine)
   - 5.6 [LLM-Powered Deep Analysis](#56-llm-powered-deep-analysis)
6. [ML Model Training Pipeline](#6-ml-model-training-pipeline)
7. [Input / Output Specification](#7-input--output-specification)
8. [Integration with Orchestrator](#8-integration-with-orchestrator)
9. [Decision Thresholds & Scoring](#9-decision-thresholds--scoring)
10. [Test Scenarios & Validation](#10-test-scenarios--validation)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Overview

The **Credibility & Policy Interpretation Agent** is the fourth agent in the multi-agent insurance claim processing pipeline. It receives data from three upstream agents (Image Processing, PDF Processing, and Requirements) and performs two core functions:

1. **User Credibility Assessment** — Evaluates how trustworthy a claim is based on the claimant's history, document integrity, claim patterns, and behavioral signals. Uses a **trained scikit-learn Random Forest classifier** (92% accuracy, 0.99 AUC) operating on 11 engineered features, with automatic fallback to a weighted heuristic if the model file is unavailable.

2. **Policy Interpretation** — Validates the claim against the insurance policy's terms and conditions, including coverage verification, exclusion matching, waiting period validation, sub-limit enforcement, and co-pay determination. This is modeled as a rule-based engine simulating fine-tuned BERT/RoBERTa NLP analysis.

The agent produces a **credibility score** (0.0–1.0) and a comprehensive **policy analysis** that downstream agents (Billing, Fraud Detection) and the Orchestrator use for final claim decisioning.

---

## 2. Functional Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| FR-01 | Score user credibility on a 0–1 scale using a trained Random Forest model | Implemented |
| FR-02 | Classify credibility as HIGH / MODERATE / LOW / VERY_LOW | Implemented |
| FR-03 | Analyze claimant's past claim history for patterns and anomalies | Implemented |
| FR-04 | Verify procedure coverage against the insurance plan type | Implemented |
| FR-05 | Check all applicable exclusions (cosmetic, experimental, etc.) | Implemented |
| FR-06 | Validate waiting periods (initial, PED, specific disease, maternity) | Implemented |
| FR-07 | Enforce sub-limits per procedure category | Implemented |
| FR-08 | Determine co-pay percentage based on hospital network status | Implemented |
| FR-09 | Detect procedure–diagnosis inconsistencies | Implemented |
| FR-10 | Use LLM for deep analysis on borderline/complex cases | Implemented |
| FR-11 | Produce full reasoning trace for audit trail | Implemented |
| FR-12 | Output must be compatible with Orchestrator's decision fusion | Implemented |

---

## 3. Architecture & Design

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CredibilityAgent.process()                       │
│                                                                     │
│  ┌─────────────────┐   ┌────────────────────┐   ┌────────────────┐  │
│  │  ClaimHistory   │   │ CredibilityFeature │   │  PolicyInter-  │  │
│  │  Analyzer       │──▶│ Extractor          │   │  preter        │  │
│  │                 │   │  (11 features)     │   │                │  │
│  └─────────────────┘   └────────┬───────────┘   └───────┬────────┘  │
│                                 │                       │           │
│                                 ▼                       │           │
│                     ┌────────────────────┐              │           │
│                     │ CredibilityScoring │              │           │
│                     │ Model (Trained RF) │              │           │
│                     │ → score + rating   │              │           │
│                     └────────┬───────────┘              │           │
│                              │                          │           │
│                              ▼                          ▼           │
│                     ┌───────────────────────────────────────┐       │
│                     │           Result Aggregation          │       │
│                     │   credibility_score + policy_analysis │       │
│                     └────────────────┬──────────────────────┘       │
│                                      │                              │
│                          (if borderline/issues)                     │
│                                      ▼                              │
│                           ┌──────────────────┐                      │
│                           │  LLM Deep Policy │                      │
│                           │  Analysis (GPT)  │                      │
│                           └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Class Structure

| Class | Responsibility |
|-------|----------------|
| `CredibilityAgent` | Main agent class. Orchestrates the pipeline and produces the final result. |
| `ClaimHistoryAnalyzer` | Retrieves and analyzes a patient's historical claims for patterns. |
| `CredibilityFeatureExtractor` | Extracts 11 numerical features (normalized 0–1) from claim data. |
| `CredibilityScoringModel` | Loads a trained scikit-learn Random Forest from disk and produces credibility scores. Falls back to weighted heuristic if model file is absent. |
| `PolicyInterpreter` | Rule-based engine that validates the claim against policy terms. |

### 3.3 Knowledge Base Constants

| Constant | Purpose |
|----------|---------|
| `PLAN_PROFILES` | 3 plan types with coverage, exclusions, waiting periods, sub-limits, co-pay |
| `SPECIFIC_WAITING_PERIOD_DISEASES` | 15 diseases requiring 1–2 year waiting period |
| `PRE_EXISTING_INDICATORS` | 18 pre-existing disease keywords triggering PED wait |
| `PROCEDURE_CATEGORY_MAP` | 30+ procedures mapped to 12 medical categories |
| `NETWORK_HOSPITAL_CHAINS` | 27 recognized hospital chains in India |
| `CLAIM_HISTORY_DB` | Simulated patient history database (production: real DB) |

---

## 4. Processing Pipeline

The agent executes 6 sequential steps within `process()`:

```
Step 1: Claim History Analysis
    │   └─ Look up patient, analyze past claims, detect patterns
    ▼
Step 2: Feature Extraction
    │   └─ Extract 11 normalized features from all upstream data
    ▼
Step 3: Credibility Scoring (Trained Random Forest)
    │   └─ Load model → predict class probabilities → score + rating
    ▼
Step 4: Policy Interpretation
    │   └─ Coverage, exclusions, waiting periods, sub-limits, co-pay
    ▼
Step 5: LLM Deep Analysis (conditional)
    │   └─ Only for borderline scores (0.40–0.70) or policy issues
    ▼
Step 6: Result Assembly
        └─ Combine all outputs, compute confidence, return
```

**Execution Time:** < 0.1s without LLM, 2–5s with LLM analysis.

---

## 5. Component Deep Dive

### 5.1 Policy Knowledge Base

The knowledge base encodes Indian health insurance policy structures aligned with **IRDAI (Insurance Regulatory and Development Authority of India)** guidelines.

#### 5.1.1 Plan Profiles

Three plan types are supported:

| Plan | Covered Categories | Key Sub-Limits | Co-Pay (Network / Non-Network) |
|------|--------------------|----------------|-------------------------------|
| **Comprehensive Health** | 12 categories (cardiac, orthopedic, oncology, neurology, general, obstetric, urology, gastro, pulmonology, nephrology, ENT, ophthalmology) | Cardiac: 70% SI, Orthopedic: 60% SI, Room: 1% SI/day | 0% / 20% |
| **Basic Health** | 4 categories (cardiac, orthopedic, general, obstetric) | Cardiac: 50% SI, Orthopedic: 40% SI, Room: 0.5% SI/day | 10% / 30% |
| **Super Top-Up** | 8 categories | No sub-limits | 0% / 10% |

#### 5.1.2 Waiting Periods

Four types of waiting periods are enforced:

| Type | Comprehensive | Basic | Super Top-Up | Trigger |
|------|---------------|-------|--------------|---------|
| **Initial Waiting** | 30 days | 30 days | 30 days | All claims |
| **Pre-Existing Disease (PED)** | 730 days (2 yrs) | 1095 days (3 yrs) | 730 days (2 yrs) | Diagnosis matches PED indicators (diabetes, hypertension, etc.) |
| **Specific Disease** | 365 days (1 yr) | 730 days (2 yrs) | 365 days (1 yr) | Diagnosis matches specific disease list (hernia, cataract, etc.) |
| **Maternity** | 730 days (2 yrs) | 1095 days (3 yrs) | 730 days (2 yrs) | Delivery, cesarean, pregnancy-related |

#### 5.1.3 Standard Exclusions

8–10 exclusion categories are checked per plan:

- Cosmetic surgery
- Dental (unless accidental)
- Self-inflicted injury
- Substance abuse treatment
- Experimental treatment
- Infertility treatment
- Weight management / bariatric
- Congenital external defects
- Alternative medicine (Basic plan only)
- Non-allopathic treatments (Basic plan only)

Each exclusion is matched via keyword detection against the diagnosis and procedure strings.

#### 5.1.4 Network Hospital Detection

The system maintains a list of **27 major hospital chains** in India (Apollo, Fortis, Max, Manipal, Medanta, Narayana, etc.). Hospital names from the Image Agent are matched against this list to determine network status, which affects:
- Co-pay percentage
- Pre-authorization requirements

---

### 5.2 Claim History Analyzer

**Class:** `ClaimHistoryAnalyzer`

Analyzes the patient's historical claims to detect patterns that affect credibility.

#### Data Source
Currently uses an in-memory dictionary (`CLAIM_HISTORY_DB`) keyed by `patient_id`. In production, this would be replaced with a database query.

#### Analysis Performed

| Check | Logic | Impact on Credibility |
|-------|-------|----------------------|
| **Claim Frequency** | Count of past claims | 5+ claims: flagged as high frequency |
| **Approval Rate** | approved / total | Low rate reduces credibility |
| **Flag History** | Count of previously flagged claims | Any flags generate a WARNING |
| **Escalation Pattern** | Current amount vs. total past amounts | Current > 2x past: flagged |
| **Rejection History** | Count of rejected claims | Each rejection reduces `no_rejections` feature by 0.2 |

#### Output
```python
{
    "total_claims": int,
    "approved": int,
    "rejected": int,
    "flagged": int,
    "total_past_amount": float,
    "policy_start_date": str,
    "claim_details": List[Dict]
}
```

---

### 5.3 Feature Extraction Engine

**Class:** `CredibilityFeatureExtractor`

Extracts **11 normalized features** (all in 0.0–1.0 range) from the upstream agent data and claim history. These features simulate the input vector for a Random Forest classifier.

#### Feature Definitions

| # | Feature | Source | Normalization | RF Importance |
|---|---------|--------|---------------|---------------|
| 1 | `flag_rate` | Claim History | flagged/total (0.0 if no history) | **25.04%** |
| 2 | `no_rejections` | Claim History | 1.0 if none, -0.2 per rejection | **21.03%** |
| 3 | `image_integrity` | Image Agent | Direct score (0–1) | **15.66%** |
| 4 | `claim_approval_rate` | Claim History | approved/total (0.5 if no history) | **14.70%** |
| 5 | `policy_tenure_norm` | Claim History | days_since_start/1095, capped at 1.0 | **10.25%** |
| 6 | `claim_frequency` | Claim History | total_claims/10, capped at 1.0 | 5.88% |
| 7 | `docs_complete` | Requirements Agent | 1.0 if met, 0.3 if not | 2.00% |
| 8 | `proc_diag_consistency` | Image Agent | 1.0=match, 0.5=uncertain, 0.3=mismatch | 1.74% |
| 9 | `timing_score` | Image Agent + History | <30 days→0.3, <90 days→0.7, else→1.0 | 1.58% |
| 10 | `amount_reasonableness` | Image + PDF Agents | Tiered: ≤30% SI→1.0, ≤60%→0.85, ≤80%→0.6, >80%→0.4 | 1.39% |
| 11 | `network_hospital` | Image Agent | 1.0 if network, 0.5 if not | 0.73% |

**Feature importances are learned by the trained Random Forest model (see Section 6).**

#### Procedure–Diagnosis Consistency Check

The `_check_procedure_diagnosis_match()` method validates that the claimed procedures align with the diagnosis. It uses a mapping of 17 diagnosis keywords to their expected procedures:

- **Score 1.0** — Diagnosis found AND at least one procedure matches (e.g., "STEMI" + "Angioplasty")
- **Score 0.5** — No matching diagnosis keyword found (uncertain)
- **Score 0.3** — Diagnosis keyword found but procedures DON'T match (suspicious)

---

### 5.4 Credibility Scoring Model (Trained Random Forest)

**Class:** `CredibilityScoringModel`

Uses a **trained scikit-learn `RandomForestClassifier`** loaded from disk (`data/models/credibility_rf_model.joblib`). The model was trained on 10,000 synthetic samples and tuned via 5-fold GridSearchCV. If the model file is not found, the class automatically falls back to a weighted heuristic.

#### Model Loading

On initialization, the class attempts to load the `.joblib` model artifact:

```python
# Resolved path: <project_root>/data/models/credibility_rf_model.joblib
model = joblib.load(model_path)
```

If loading fails (file missing, import error, corrupt file), the agent logs a warning and switches to the fallback scorer — no crash, no interruption.

#### Scoring Algorithm (Real Model)

```
1. Build feature vector in the order defined by _FEATURE_ORDER (11 features)

2. model.predict(X)       → predicted class (0=VERY_LOW, 1=LOW, 2=MODERATE, 3=HIGH)
   model.predict_proba(X) → class probabilities [P(0), P(1), P(2), P(3)]

3. Continuous score from weighted class probabilities:
     score = P(VERY_LOW)×0.0 + P(LOW)×0.33 + P(MODERATE)×0.67 + P(HIGH)×1.0

4. Rating = class label from the predicted class
```

This produces a smooth 0–1 score that reflects the model's confidence distribution across all four classes, rather than a hard classification.

#### Trained Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.8% |
| **Precision** (weighted) | 91.8% |
| **Recall** (weighted) | 91.8% |
| **F1 Score** (weighted) | 91.8% |
| **AUC (One-vs-Rest)** | 98.97% |
| **5-Fold CV Accuracy** | 92.2% (+/- 0.25%) |

#### Best Hyperparameters (via GridSearchCV)

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 16 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 1 |
| `class_weight` | balanced |

#### Confusion Matrix (Test Set, n=2000)

|  | Predicted VERY_LOW | Predicted LOW | Predicted MODERATE | Predicted HIGH |
|---|---|---|---|---|
| **Actual VERY_LOW** | **266** | 34 | 0 | 0 |
| **Actual LOW** | 24 | **355** | 21 | 0 |
| **Actual MODERATE** | 0 | 11 | **449** | 40 |
| **Actual HIGH** | 0 | 0 | 34 | **766** |

Key observations:
- HIGH class has 95.8% precision — the model is very reliable when it says a claimant is trustworthy
- VERY_LOW class has 91.7% precision — low false positive rate for flagging suspicious claims
- Most misclassifications are between adjacent classes (e.g., MODERATE vs HIGH), not extreme errors

#### Rating Thresholds

The rating is determined by the Random Forest's predicted class:

| Predicted Class | Rating | Orchestrator Action |
|-----------------|--------|---------------------|
| 3 | **HIGH** | Approve (if other agents pass) |
| 2 | **MODERATE** | Standard processing |
| 1 | **LOW** | Review recommended |
| 0 | **VERY_LOW** | Likely REJECT (score < 0.40 threshold) |

#### Feature Contribution Tracking

The model returns per-feature contributions computed from the Random Forest's learned feature importances:

```python
contributions = {
    "no_rejections": 0.2103,       # importance × feature_value
    "image_integrity": 0.1475,
    "claim_approval_rate": 0.0735,
    ...
}
```

This enables **model explainability** — the reasoning trace shows which features most influenced the prediction.

#### Fallback Scoring (When Model Unavailable)

If the `.joblib` file is missing, a deterministic weighted heuristic is used:

```
1. raw_score = Σ(feature_value × weight) / Σ(weights)
2. score = sigmoid(raw_score)    where sigmoid(x) = 1 / (1 + e^(-6(x-0.5)))
```

The fallback weights mirror the training data's design priorities. This ensures the agent always produces a valid score.

---

### 5.5 Policy Interpretation Engine

**Class:** `PolicyInterpreter`

Performs 7 sequential policy validation checks:

#### Check 1: Plan Type Identification

Matches the coverage type string from the PDF Agent to one of three plan profiles:
- Keywords "comprehensive" or "individual" → `comprehensive_health`
- Keywords "top-up" or "super" → `super_top_up`
- Keyword "basic" → `basic_health`
- Default → `comprehensive_health`

#### Check 2: Coverage Verification

Maps claimed procedures to medical categories using `PROCEDURE_CATEGORY_MAP` (30+ entries), then verifies each category is covered under the identified plan.

Example:
```
Procedures: ["Coronary Angioplasty", "Stent Placement"]
  → Categories: ["cardiac"]
  → Comprehensive Health covers "cardiac"? YES
```

#### Check 3: Exclusion Matching

Scans the diagnosis and procedure strings against exclusion keyword lists. Each of the 8–10 exclusion types has associated keywords:

| Exclusion | Keywords Checked |
|-----------|------------------|
| Cosmetic Surgery | cosmetic, aesthetic, liposuction, rhinoplasty, botox |
| Self-Inflicted | self-inflicted, self inflicted, suicide attempt |
| Substance Abuse | alcohol rehabilitation, drug rehabilitation, substance abuse |
| Experimental | experimental, clinical trial, investigational |
| Infertility | ivf, iui, infertility, fertility treatment |
| Weight Management | bariatric, weight loss surgery, liposuction |
| Alternative Medicine | ayurveda, homeopathy, naturopathy, unani |

#### Check 4: Waiting Period Validation

Calculates `days_since_policy_start = admission_date - policy_start_date` and validates against applicable waiting periods in priority order:

1. **Initial Waiting** (30 days) — applies to ALL claims
2. **Pre-Existing Disease** (730–1095 days) — if diagnosis contains PED indicators
3. **Specific Disease** (365–730 days) — if diagnosis/procedure matches listed diseases
4. **Maternity** (730–1095 days) — if delivery/pregnancy related

Returns `satisfied: True/False` with detailed explanation.

#### Check 5: Sum Insured Verification

Simple comparison: `total_bill ≤ sum_insured`

#### Check 6: Sub-Limit Enforcement

Looks up the most restrictive applicable sub-limit for the procedure category:

```
Example for Cardiac procedure, Comprehensive plan, SI = ₹10,00,000:
  Sub-limit = 70% × ₹10,00,000 = ₹7,00,000
  Claim ₹4,85,000 ≤ ₹7,00,000 → Within sub-limit
```

#### Check 7: Co-Pay & Pre-Authorization

Determined by hospital network status:

| | Network Hospital | Non-Network Hospital |
|---|---|---|
| Comprehensive | 0% co-pay, No pre-auth needed | 20% co-pay, Pre-auth required |
| Basic | 10% co-pay | 30% co-pay |
| Super Top-Up | 0% co-pay | 10% co-pay |

#### Output Structure
```python
{
    "procedure_covered": bool,
    "within_sum_insured": bool,
    "sub_limit_applicable": bool,
    "sub_limit_amount": int,
    "claim_within_sub_limit": bool,
    "waiting_period_satisfied": bool,
    "exclusions_applicable": bool,
    "exclusion_detail": Optional[str],
    "co_pay_percentage": int,
    "pre_authorization": str,
    "plan_type": str,
    "is_network_hospital": bool,
}
```

---

### 5.6 LLM-Powered Deep Analysis

The agent optionally invokes the LLM (GPT-4o-mini) for deep policy analysis when:

1. **Borderline credibility score** — Score between 0.40 and 0.70 (neither clearly credible nor clearly suspicious)
2. **Policy issues detected** — Procedure not covered, exclusions applicable, or waiting period not satisfied

#### LLM Prompt Structure

The LLM receives a structured prompt containing:
- Patient details (name, diagnosis, procedures, hospital)
- Financial data (total bill, sum insured)
- Policy context (coverage type, policy start date)
- Current assessment results (coverage, exclusions, waiting period, co-pay)
- Credibility score and past claim count

#### System Prompt
```
You are an insurance policy interpretation AI specializing in Indian
health insurance regulations (IRDAI guidelines). Analyze claims against
policy terms and provide professional assessments. Respond in valid JSON only.
```

#### Expected LLM Response
```json
{
    "summary": "1-2 sentence assessment",
    "risk_flags": ["list", "of", "concerns"],
    "recommendation": "APPROVE / REVIEW / REJECT",
    "confidence": 0.85
}
```

#### Graceful Degradation
If the LLM is unavailable or fails, the agent continues with rule-based analysis only. LLM analysis is an enhancement, not a dependency.

---

## 6. ML Model Training Pipeline

The credibility model is trained using a dedicated script at `src/models/train_credibility_model.py`.

### 6.1 How to Retrain

```bash
# From project root
python -m src.models.train_credibility_model
```

This generates:
- `data/models/credibility_rf_model.joblib` — serialized model
- `data/models/credibility_model_metadata.json` — metrics, params, feature importances

The agent automatically picks up the new model on next initialization.

### 6.2 Synthetic Data Generation

Since real labeled claim data is not yet available, the training pipeline generates **10,000 synthetic samples** with realistic feature distributions across 4 claimant profiles:

| Profile | Fraction | Characteristics |
|---------|----------|-----------------|
| **HIGH** (label 3) | 40% | Long tenure, clean history, network hospitals, high image integrity, matching procedures |
| **MODERATE** (label 2) | 25% | Mixed signals, some minor flags, moderate tenure |
| **LOW** (label 1) | 20% | New policies, non-network hospitals, some inconsistencies, incomplete docs |
| **VERY_LOW** (label 0) | 15% | Flagged history, low image integrity, procedure mismatches, very new policies |

Each feature is sampled from profile-specific **Beta distributions** (for continuous features) or **categorical distributions** (for discrete features), ensuring realistic correlations between features within each credibility class.

### 6.3 Training Process

1. **Train/Test Split** — 80/20 stratified split
2. **Hyperparameter Tuning** — 5-fold `GridSearchCV` over:
   - `n_estimators`: [100, 200, 300]
   - `max_depth`: [8, 12, 16, None]
   - `min_samples_split`: [2, 5, 10]
   - `min_samples_leaf`: [1, 2, 4]
   - `class_weight`: [balanced, balanced_subsample]
3. **Evaluation** — Accuracy, precision, recall, F1, AUC (one-vs-rest), confusion matrix
4. **Artifact Saving** — Model + metadata to `data/models/`

### 6.4 Output Artifacts

| File | Format | Contents |
|------|--------|----------|
| `credibility_rf_model.joblib` | Serialized sklearn model | Trained `RandomForestClassifier` |
| `credibility_model_metadata.json` | JSON | Feature names, label map, best params, metrics, feature importances, confusion matrix |

### 6.5 Replacing with Real Data

When real labeled claim data becomes available:

1. Replace `generate_synthetic_data()` with a data loader that reads from a CSV/database
2. Ensure the DataFrame has the same 11 feature columns + `label` column (0–3)
3. Run `python -m src.models.train_credibility_model`
4. The agent will automatically use the newly trained model

No code changes are needed in `credibility_agent.py` — the model interface is the same regardless of training data source.

---

## 7. Input / Output Specification

### 6.1 Input

The agent receives `claim_data: Dict` containing results from upstream agents:

```python
{
    "image": {
        "status": "success",
        "confidence": 0.94,
        "output": {
            "patient_name": str,
            "patient_id": str,          # Used for history lookup
            "hospital": str,            # Used for network detection
            "admission_date": str,      # YYYY-MM-DD
            "discharge_date": str,
            "diagnosis": str,           # Used for PED, exclusions, consistency
            "procedures": List[str],    # Used for coverage, consistency
            "total_bill_amount": float, # Used for amount reasonableness
            "image_integrity": {
                "score": float          # Used as feature
            }
        }
    },
    "pdf": {
        "status": "success",
        "output": {
            "sum_insured": float,       # Used for sub-limits, amount check
            "coverage_type": str,       # Used for plan identification
        }
    },
    "requirements": {
        "status": "success",
        "output": {
            "requirements_met": bool    # Used as feature
        }
    }
}
```

### 6.2 Output

```python
{
    "agent": "Credibility & Policy Interpretation Agent",
    "owner": "Shruti Roy",
    "status": "success",
    "reasoning": List[str],            # Full audit trail (20-30 entries)
    "output": {
        "credibility_score": float,    # 0.0-1.0 (KEY: used by orchestrator)
        "credibility_rating": str,     # HIGH / MODERATE / LOW / VERY_LOW
        "claim_history": {
            "total_claims": int,
            "approved": int,
            "rejected": int,
            "flagged": int,
        },
        "policy_analysis": {           # KEY: used by orchestrator + billing agent
            "procedure_covered": bool,
            "within_sum_insured": bool,
            "sub_limit_applicable": bool,
            "sub_limit_amount": int,
            "claim_within_sub_limit": bool,
            "waiting_period_satisfied": bool,
            "exclusions_applicable": bool,
            "exclusion_detail": Optional[str],
            "co_pay_percentage": int,
            "pre_authorization": str,
            "plan_type": str,
            "is_network_hospital": bool,
        },
        "feature_scores": Dict[str, float],        # All 11 features
        "feature_contributions": Dict[str, float],  # Per-feature model contribution
        "ml_model": str,                            # Model description
        "nlp_model": str,                           # NLP model description
        "llm_analysis": Optional[Dict],             # LLM output (if triggered)
    },
    "confidence": float,               # Agent self-confidence (0.50-0.99)
    "processing_time": str,            # e.g., "0.1s"
}
```

---

## 8. Integration with Orchestrator

The Orchestrator (`src/agents/orchestrator.py`) consumes the following fields from this agent:

### 8.1 Decision Fusion

| Orchestrator Field | Source Path | Usage |
|-------------------|-------------|-------|
| `cred_score` | `output.credibility_score` | If < 0.40 → **REJECT** claim |
| `confidence` | `result.confidence` | Weight: **0.20** in weighted confidence |

### 8.2 Claim Summary

| Summary Field | Source Path |
|---------------|-------------|
| `credibility_score` | `output.credibility_score` |

### 8.3 Downstream Agents

The **Billing Agent** reads `output.policy_analysis` for:
- `sub_limit_amount` — to cap approved billing
- `co_pay_percentage` — to apply co-pay deductions
- `procedure_covered` — to reject non-covered procedures
- `exclusions_applicable` — to reject excluded claims
- `waiting_period_satisfied` — to reject claims in waiting period

### 8.4 Pipeline Position

```
Image Agent → PDF Agent → Requirements Agent → [CREDIBILITY AGENT] → Billing Agent → Fraud Agent → Orchestrator
```

The credibility agent is **4th** in the pipeline and receives data from the first 3 agents.

---

## 9. Decision Thresholds & Scoring

### 9.1 Orchestrator Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| `credibility_reject` | 0.40 | Score below this → REJECT |

### 9.2 Agent Confidence Computation

Base confidence starts at **0.92** and is adjusted:

| Condition | Adjustment |
|-----------|------------|
| Clear-cut score (>0.80 or <0.30) | +0.03 |
| Procedure not covered | -0.05 |
| Exclusions applicable | -0.05 |
| Waiting period not satisfied | -0.03 |
| Low procedure-diagnosis consistency (<0.5) | -0.05 |

Final confidence is clamped to [0.50, 0.99].

---

## 10. Test Scenarios & Validation

### 10.1 Normal Case (Approved)

**Input:** Rajesh Kumar, Apollo Hyderabad, STEMI + Angioplasty + Stent, ₹4.85L, Comprehensive plan, SI ₹10L, policy since Apr 2024.

| Check | Result |
|-------|--------|
| Credibility Score | **0.991 (HIGH)** |
| Agent Confidence | 0.95 |
| Coverage | Cardiac — covered |
| Exclusions | None |
| Waiting Period | 289 days — satisfied |
| Sub-Limit | ₹7L — within limit |
| Co-Pay | 0% (Network hospital) |
| Top Contributors | no_rejections (0.2103), image_integrity (0.1475), claim_approval_rate (0.0735) |

### 10.2 Risky Case (Borderline)

**Input:** Unknown clinic Delhi, Diabetes + Dialysis, ₹8L, Basic plan, SI ₹10L, policy since Jan 2025, 2 flagged past claims.

| Check | Result |
|-------|--------|
| Credibility Score | **0.675 (MODERATE)** |
| Agent Confidence | 0.84 |
| Coverage | Nephrology — NOT covered under Basic |
| Exclusions | None |
| Waiting Period | 14 days — NOT satisfied (need 30) |
| Co-Pay | 30% (Non-network) |

Note: Even though the credibility score is MODERATE, the policy interpretation catches the coverage and waiting period failures. The orchestrator combines both signals — the claim would be HOLD or REJECT based on the policy analysis.

### 10.3 Edge Cases Handled

| Scenario | Behavior |
|----------|----------|
| Unknown patient ID | Falls back to DEFAULT history (no claims) |
| Missing dates | Assumes waiting period satisfied |
| No procedures listed | Defaults to "general" category |
| No LLM client available | Skips LLM analysis gracefully |
| Unmapped procedure | Not matched to any category — defaults to "general" |
| Multiple procedure categories | Checks ALL categories against plan coverage |

---

## 11. Future Enhancements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Train on Real Data** | Replace synthetic data generator with real labeled claim data for higher model accuracy | High |
| **Fine-tuned BERT** | Deploy a BERT/RoBERTa model fine-tuned on insurance policy corpus for clause interpretation | High |
| **Database Integration** | Replace `CLAIM_HISTORY_DB` with PostgreSQL/MongoDB queries | High |
| **RAG for Policy Clauses** | Use FAISS/Pinecone vector DB with the PDF Agent's indexed chunks for semantic policy search | Medium |
| **More Plan Types** | Add Family Floater, Group Health, Critical Illness, Personal Accident plans | Medium |
| **Regional Pricing** | Integrate city-level pricing benchmarks for amount reasonableness | Medium |
| **Model Monitoring** | Track prediction drift and feature distribution changes in production | Medium |
| **Real-Time Model Retraining** | Feedback loop from adjuster decisions to retrain the credibility model | Low |
| **Multi-Language Support** | Support policy documents in Hindi, Tamil, Telugu, etc. | Low |

---

## Appendix A: Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Async Runtime | asyncio |
| ML Model | scikit-learn `RandomForestClassifier` (v1.6.1), trained on 10K samples |
| Model Serialization | joblib (v1.5.3) |
| Training Data | numpy (v2.0.2), pandas (v2.3.3) |
| Hyperparameter Tuning | scikit-learn `GridSearchCV` (5-fold CV) |
| NLP / Policy Engine | Rule-based keyword matching (future: fine-tuned BERT/RoBERTa) |
| LLM Integration | OpenAI GPT-4o-mini via async client |
| Logging | Python `logging` module |
| Configuration | YAML (`config/prompt_templates.yaml`) |

## Appendix B: File References

| File | Purpose |
|------|---------|
| `src/agents/credibility_agent.py` | Main agent implementation |
| `src/models/train_credibility_model.py` | Model training pipeline (data generation + GridSearchCV + evaluation) |
| `src/models/__init__.py` | ML models package init |
| `data/models/credibility_rf_model.joblib` | Trained Random Forest model artifact |
| `data/models/credibility_model_metadata.json` | Model metadata (params, metrics, feature importances) |
| `config/prompt_templates.yaml` | LLM prompt templates (`credibility_policy` section) |
| `config/model_config.yaml` | Agent timeout (45s), decision thresholds |
| `src/agents/orchestrator.py` | Consumes credibility output for decision fusion |
| `src/agents/billing_agent.py` | Consumes `policy_analysis` for billing limits |

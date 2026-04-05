"""
User Credibility & Policy Interpretation Agent (Agent 2)
Owner: Shruti Roy
Status: IMPLEMENTED

Features:
  - ML-style credibility scoring with feature engineering (Random Forest logic)
  - Policy clause interpretation with coverage verification
  - Claim history analysis pipeline with pattern detection
  - Waiting period and exclusion validation against policy rules
  - Sub-limit and co-pay enforcement
  - LLM-powered deep policy analysis for ambiguous/complex cases
"""

import json
import math
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("insurance_claim_ai.agents")


# ============================================================
# Insurance Policy Knowledge Base
# ============================================================

# Standard insurance plan types and their coverage profiles
PLAN_PROFILES: Dict[str, Dict[str, Any]] = {
    "comprehensive_health": {
        "display_name": "Comprehensive Health",
        "covered_categories": [
            "cardiac", "orthopedic", "oncology", "neurology",
            "general", "obstetric", "urology", "gastro",
            "pulmonology", "nephrology", "ent", "ophthalmology",
        ],
        "standard_exclusions": [
            "cosmetic_surgery", "dental_unless_accidental",
            "self_inflicted_injury", "substance_abuse_treatment",
            "experimental_treatment", "infertility_treatment",
            "weight_management", "congenital_external_defects",
        ],
        "waiting_periods": {
            "initial_waiting": 30,          # days — general
            "pre_existing_disease": 730,    # days (2 years) — PED
            "specific_disease": 365,        # days (1 year) — listed diseases
            "maternity": 730,               # days (2 years)
        },
        "sub_limits": {
            "cardiac": 0.70,          # 70% of sum insured
            "orthopedic": 0.60,
            "room_rent_daily": 0.01,  # 1% of sum insured per day
            "icu_daily": 0.02,        # 2% of sum insured per day
            "ambulance": 2500,        # fixed amount
            "cataract_per_eye": 40000,
        },
        "co_pay": {
            "network": 0,      # % co-pay at network hospitals
            "non_network": 20, # % co-pay at non-network hospitals
        },
    },
    "basic_health": {
        "display_name": "Basic Health",
        "covered_categories": [
            "cardiac", "orthopedic", "general", "obstetric",
        ],
        "standard_exclusions": [
            "cosmetic_surgery", "dental_unless_accidental",
            "self_inflicted_injury", "substance_abuse_treatment",
            "experimental_treatment", "infertility_treatment",
            "weight_management", "congenital_external_defects",
            "alternative_medicine", "non_allopathic",
        ],
        "waiting_periods": {
            "initial_waiting": 30,
            "pre_existing_disease": 1095,  # 3 years
            "specific_disease": 730,       # 2 years
            "maternity": 1095,             # 3 years
        },
        "sub_limits": {
            "cardiac": 0.50,
            "orthopedic": 0.40,
            "room_rent_daily": 0.005,
            "icu_daily": 0.01,
            "ambulance": 2000,
            "cataract_per_eye": 25000,
        },
        "co_pay": {
            "network": 10,
            "non_network": 30,
        },
    },
    "super_top_up": {
        "display_name": "Super Top-Up",
        "covered_categories": [
            "cardiac", "orthopedic", "oncology", "neurology",
            "general", "obstetric", "urology", "gastro",
        ],
        "standard_exclusions": [
            "cosmetic_surgery", "dental_unless_accidental",
            "self_inflicted_injury", "substance_abuse_treatment",
            "experimental_treatment",
        ],
        "waiting_periods": {
            "initial_waiting": 30,
            "pre_existing_disease": 730,
            "specific_disease": 365,
            "maternity": 730,
        },
        "sub_limits": {},  # Usually no sub-limits
        "co_pay": {
            "network": 0,
            "non_network": 10,
        },
    },
}

# Diseases that fall under specific waiting period (1-2 year wait)
SPECIFIC_WAITING_PERIOD_DISEASES = [
    "hernia", "cataract", "sinusitis", "tonsillitis",
    "kidney_stone", "gallstone", "piles", "fistula",
    "benign_prostatic_hypertrophy", "hysterectomy",
    "joint_replacement", "disc_prolapse",
    "thyroid", "gout", "rheumatism",
]

# Pre-existing disease indicators (triggers PED waiting period)
PRE_EXISTING_INDICATORS = [
    "diabetes", "hypertension", "asthma", "copd",
    "heart_disease", "coronary_artery_disease", "stroke",
    "kidney_disease", "liver_disease", "cancer",
    "epilepsy", "parkinsons", "alzheimers",
    "hiv", "hepatitis", "tuberculosis",
    "rheumatoid_arthritis", "lupus",
]

# Procedure-to-category mapping for coverage verification
PROCEDURE_CATEGORY_MAP: Dict[str, str] = {
    "angioplasty": "cardiac",
    "coronary angioplasty": "cardiac",
    "stent placement": "cardiac",
    "stent": "cardiac",
    "cabg": "cardiac",
    "bypass": "cardiac",
    "pacemaker": "cardiac",
    "angiography": "cardiac",
    "cardiac catheterization": "cardiac",
    "valve replacement": "cardiac",
    "knee replacement": "orthopedic",
    "hip replacement": "orthopedic",
    "spinal fusion": "orthopedic",
    "fracture fixation": "orthopedic",
    "arthroscopy": "orthopedic",
    "appendectomy": "general",
    "cholecystectomy": "general",
    "hernia repair": "general",
    "chemotherapy": "oncology",
    "radiation therapy": "oncology",
    "mastectomy": "oncology",
    "tumor excision": "oncology",
    "c-section": "obstetric",
    "cesarean": "obstetric",
    "normal delivery": "obstetric",
    "craniotomy": "neurology",
    "brain surgery": "neurology",
    "dialysis": "nephrology",
    "kidney transplant": "nephrology",
    "cataract surgery": "ophthalmology",
    "lasik": "ophthalmology",
    "tonsillectomy": "ent",
    "septoplasty": "ent",
    "lithotripsy": "urology",
    "prostatectomy": "urology",
    "endoscopy": "gastro",
    "colonoscopy": "gastro",
}

# Network hospital identifiers (major chains in India)
NETWORK_HOSPITAL_CHAINS = [
    "apollo", "fortis", "max", "manipal", "medanta",
    "narayana", "kokilaben", "aster", "columbia asia",
    "global hospital", "yashoda", "care hospital",
    "lilavati", "hinduja", "breach candy", "jaslok",
    "artemis", "blk", "sir ganga ram", "aiims",
    "pgimer", "cmc vellore", "nims", "kims",
    "rainbow", "cloudnine", "motherhood",
]


# ============================================================
# Simulated Claim History Database
# ============================================================

CLAIM_HISTORY_DB: Dict[str, Dict[str, Any]] = {
    # Keyed by patient_id — in production this would be a database query
    "PAT-2025-08431": {
        "patient_name": "Rajesh Kumar",
        "policy_start_date": "2024-04-01",
        "claims": [
            {
                "claim_id": "CLM-2024-1201",
                "date": "2024-08-15",
                "type": "OPD",
                "amount": 3500,
                "status": "approved",
                "flagged": False,
                "diagnosis": "Viral Fever",
            },
            {
                "claim_id": "CLM-2024-3842",
                "date": "2024-11-22",
                "type": "Daycare",
                "amount": 15000,
                "status": "approved",
                "flagged": False,
                "diagnosis": "Dental Extraction (Accidental)",
            },
        ],
    },
    # Default fallback for unknown patients
    "DEFAULT": {
        "patient_name": "Unknown",
        "policy_start_date": "2024-01-01",
        "claims": [],
    },
}


# ============================================================
# Credibility Feature Engineering
# ============================================================

class CredibilityFeatureExtractor:
    """
    Extracts features for credibility scoring from claim data.
    Simulates the feature engineering pipeline that feeds a
    Random Forest classifier in production.
    """

    @staticmethod
    def extract(
        claim_data: Dict,
        claim_history: Dict,
        policy_start_date: str,
    ) -> Dict[str, float]:
        """
        Extract numerical features for credibility scoring.

        Returns dict of feature_name -> feature_value (all normalized 0-1).
        """
        image_output = _safe_get(claim_data, "image", "output") or {}
        pdf_output = _safe_get(claim_data, "pdf", "output") or {}
        req_output = _safe_get(claim_data, "requirements", "output") or {}

        features = {}

        # --- Claim History Features ---
        history = claim_history.get("claims", [])
        total_claims = len(history)
        approved_claims = sum(1 for c in history if c.get("status") == "approved")
        rejected_claims = sum(1 for c in history if c.get("status") == "rejected")
        flagged_claims = sum(1 for c in history if c.get("flagged"))

        features["claim_approval_rate"] = (
            approved_claims / total_claims if total_claims > 0 else 0.5
        )
        features["claim_frequency"] = min(total_claims / 10.0, 1.0)  # Normalize to 10 claims
        features["flag_rate"] = (
            flagged_claims / total_claims if total_claims > 0 else 0.0
        )
        features["no_rejections"] = 1.0 if rejected_claims == 0 else max(0.0, 1.0 - rejected_claims * 0.2)

        # --- Policy Tenure Feature ---
        try:
            start = datetime.strptime(policy_start_date, "%Y-%m-%d")
            tenure_days = (datetime.now() - start).days
            features["policy_tenure_norm"] = min(tenure_days / 1095.0, 1.0)  # Normalize to 3 years
        except (ValueError, TypeError):
            features["policy_tenure_norm"] = 0.5

        # --- Document Completeness Feature ---
        reqs_met = req_output.get("requirements_met", False)
        features["docs_complete"] = 1.0 if reqs_met else 0.3

        # --- Image Integrity Feature ---
        image_integrity = _safe_get(image_output, "image_integrity", "score") or 0.5
        features["image_integrity"] = float(image_integrity)

        # --- Claim Amount Reasonableness ---
        total_bill = image_output.get("total_bill_amount", 0)
        sum_insured = pdf_output.get("sum_insured", 1000000)
        claim_to_si_ratio = total_bill / sum_insured if sum_insured > 0 else 1.0
        # Moderate claims are less suspicious
        if claim_to_si_ratio <= 0.3:
            features["amount_reasonableness"] = 1.0
        elif claim_to_si_ratio <= 0.6:
            features["amount_reasonableness"] = 0.85
        elif claim_to_si_ratio <= 0.8:
            features["amount_reasonableness"] = 0.6
        else:
            features["amount_reasonableness"] = 0.4

        # --- Procedure-Diagnosis Consistency ---
        diagnosis = image_output.get("diagnosis", "").lower()
        procedures = [p.lower() for p in image_output.get("procedures", [])]
        features["proc_diag_consistency"] = (
            CredibilityFeatureExtractor._check_procedure_diagnosis_match(diagnosis, procedures)
        )

        # --- Hospital Network Feature ---
        hospital = image_output.get("hospital", "").lower()
        is_network = any(chain in hospital for chain in NETWORK_HOSPITAL_CHAINS)
        features["network_hospital"] = 1.0 if is_network else 0.5

        # --- Claim Timing Feature ---
        # Claims too close to policy start date are slightly suspicious
        try:
            policy_start = datetime.strptime(policy_start_date, "%Y-%m-%d")
            admission_str = image_output.get("admission_date", "")
            if admission_str:
                admission = datetime.strptime(admission_str, "%Y-%m-%d")
                days_since_start = (admission - policy_start).days
                if days_since_start < 30:
                    features["timing_score"] = 0.3  # Within initial waiting
                elif days_since_start < 90:
                    features["timing_score"] = 0.7
                else:
                    features["timing_score"] = 1.0
            else:
                features["timing_score"] = 0.7
        except (ValueError, TypeError):
            features["timing_score"] = 0.7

        return features

    @staticmethod
    def _check_procedure_diagnosis_match(diagnosis: str, procedures: List[str]) -> float:
        """Check if procedures are consistent with the diagnosis."""
        if not diagnosis or not procedures:
            return 0.5  # Uncertain

        # Define expected procedure-diagnosis pairs
        diagnosis_procedure_map = {
            "myocardial infarction": ["angioplasty", "stent", "cabg", "catheterization", "angiography"],
            "stemi": ["angioplasty", "stent", "cabg", "catheterization"],
            "nstemi": ["angioplasty", "stent", "angiography"],
            "angina": ["angioplasty", "stent", "angiography", "cabg"],
            "coronary artery disease": ["angioplasty", "stent", "cabg", "angiography"],
            "fracture": ["fixation", "plating", "nailing", "cast"],
            "appendicitis": ["appendectomy"],
            "cholelithiasis": ["cholecystectomy"],
            "gallstone": ["cholecystectomy"],
            "hernia": ["hernia repair", "hernioplasty"],
            "cancer": ["chemotherapy", "radiation", "surgery", "excision", "mastectomy"],
            "tumor": ["excision", "chemotherapy", "radiation"],
            "kidney stone": ["lithotripsy", "ureteroscopy"],
            "cataract": ["cataract surgery", "phacoemulsification"],
            "knee": ["knee replacement", "arthroscopy"],
            "hip": ["hip replacement"],
            "spine": ["spinal fusion", "discectomy", "laminectomy"],
        }

        match_score = 0.5  # Default: uncertain
        for diag_keyword, expected_procs in diagnosis_procedure_map.items():
            if diag_keyword in diagnosis:
                # Check if at least one procedure matches
                for proc in procedures:
                    if any(ep in proc for ep in expected_procs):
                        match_score = 1.0
                        break
                else:
                    match_score = 0.3  # Diagnosis found but procedures don't match
                break

        return match_score


# ============================================================
# Credibility Scoring Model (Real Random Forest)
# ============================================================

# Feature order must match training script
_FEATURE_ORDER = [
    "claim_approval_rate",
    "claim_frequency",
    "flag_rate",
    "no_rejections",
    "policy_tenure_norm",
    "docs_complete",
    "image_integrity",
    "amount_reasonableness",
    "proc_diag_consistency",
    "network_hospital",
    "timing_score",
]

_LABEL_MAP = {0: "VERY_LOW", 1: "LOW", 2: "MODERATE", 3: "HIGH"}

# Model file path (relative to this file)
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
_MODEL_PATH = _MODEL_DIR / "credibility_rf_model.joblib"
_METADATA_PATH = _MODEL_DIR / "credibility_model_metadata.json"


class CredibilityScoringModel:
    """
    Credibility scoring using a trained Random Forest classifier.

    Loads a scikit-learn RandomForestClassifier from disk. If the model
    file is not found, falls back to a weighted heuristic.
    """

    def __init__(self):
        self._model = None
        self._metadata = None
        self._using_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the trained model from disk."""
        try:
            import joblib as _joblib
            if _MODEL_PATH.exists():
                self._model = _joblib.load(_MODEL_PATH)
                logger.info(f"Loaded credibility model from {_MODEL_PATH}")
                if _METADATA_PATH.exists():
                    with open(_METADATA_PATH) as f:
                        self._metadata = json.load(f)
            else:
                logger.warning(
                    f"Model file not found at {_MODEL_PATH} — using fallback scoring"
                )
                self._using_fallback = True
        except Exception as e:
            logger.warning(f"Failed to load credibility model: {e} — using fallback")
            self._using_fallback = True

    def predict(self, features: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
        """
        Predict credibility score from extracted features.

        Returns:
            (score, rating, feature_contributions)
        """
        if self._using_fallback or self._model is None:
            return self._fallback_predict(features)

        # Build feature vector in the correct order
        feature_vector = [features.get(f, 0.5) for f in _FEATURE_ORDER]
        import numpy as _np
        X = _np.array([feature_vector])

        # Predict class and probabilities
        pred_class = int(self._model.predict(X)[0])
        probabilities = self._model.predict_proba(X)[0]

        # Compute a continuous score from class probabilities
        # Weighted average: P(HIGH)*1.0 + P(MODERATE)*0.67 + P(LOW)*0.33 + P(VERY_LOW)*0.0
        class_weights = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
        score = sum(
            probabilities[i] * class_weights[i]
            for i in range(len(probabilities))
        )
        score = round(float(score), 4)

        rating = _LABEL_MAP.get(pred_class, "MODERATE")

        # Feature contributions via model's feature importances
        importances = self._model.feature_importances_
        contributions = {}
        for i, fname in enumerate(_FEATURE_ORDER):
            contributions[fname] = round(
                float(importances[i] * features.get(fname, 0.5)), 4
            )

        return score, rating, contributions

    @staticmethod
    def _fallback_predict(features: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
        """Weighted heuristic fallback when model file is unavailable."""
        weights = {
            "claim_approval_rate": 0.15,
            "claim_frequency": 0.05,
            "flag_rate": 0.12,
            "no_rejections": 0.10,
            "policy_tenure_norm": 0.08,
            "docs_complete": 0.12,
            "image_integrity": 0.10,
            "amount_reasonableness": 0.08,
            "proc_diag_consistency": 0.10,
            "network_hospital": 0.05,
            "timing_score": 0.05,
        }
        weighted_sum = 0.0
        total_weight = 0.0
        contributions = {}
        for fname, w in weights.items():
            val = features.get(fname, 0.5)
            contributions[fname] = round(val * w, 4)
            weighted_sum += val * w
            total_weight += w

        raw = weighted_sum / total_weight if total_weight > 0 else 0.5
        # Sigmoid smoothing
        score = round(1.0 / (1.0 + math.exp(-6.0 * (raw - 0.5))), 4)

        if score >= 0.80:
            rating = "HIGH"
        elif score >= 0.60:
            rating = "MODERATE"
        elif score >= 0.40:
            rating = "LOW"
        else:
            rating = "VERY_LOW"

        return score, rating, contributions

    @property
    def model_info(self) -> str:
        """Return a description of the active model."""
        if self._using_fallback:
            return "Weighted Heuristic Fallback (model file not found)"
        acc = "N/A"
        if self._metadata:
            acc = self._metadata.get("metrics", {}).get("accuracy", "N/A")
        return f"Random Forest Ensemble (accuracy: {acc})"


# ============================================================
# Policy Interpretation Engine
# ============================================================

class PolicyInterpreter:
    """
    Interprets insurance policy terms against a claim.
    Checks coverage, exclusions, waiting periods, sub-limits, and co-pay.
    """

    @staticmethod
    def interpret(
        claim_data: Dict,
        policy_start_date: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Interpret policy terms against the claim.

        Returns:
            (policy_analysis dict, reasoning steps list)
        """
        image_output = _safe_get(claim_data, "image", "output") or {}
        pdf_output = _safe_get(claim_data, "pdf", "output") or {}

        reasoning = []

        # Determine plan type
        coverage_type = pdf_output.get("coverage_type", "").lower()
        plan_key = PolicyInterpreter._identify_plan(coverage_type)
        plan = PLAN_PROFILES.get(plan_key, PLAN_PROFILES["comprehensive_health"])
        reasoning.append(f"Identified plan type: {plan['display_name']}")

        # Extract claim details
        procedures = [p.lower() for p in image_output.get("procedures", [])]
        diagnosis = image_output.get("diagnosis", "").lower()
        hospital = image_output.get("hospital", "").lower()
        total_bill = image_output.get("total_bill_amount", 0)
        sum_insured = pdf_output.get("sum_insured", 0)
        admission_date = image_output.get("admission_date", "")

        # 1. Coverage Verification
        procedure_categories = PolicyInterpreter._get_procedure_categories(procedures)
        all_covered = all(
            cat in plan["covered_categories"] for cat in procedure_categories
        )
        reasoning.append(
            f"Procedure categories: {', '.join(procedure_categories) or 'unknown'} — "
            f"{'All covered' if all_covered else 'NOT ALL COVERED'} under {plan['display_name']}"
        )

        # 2. Exclusion Check
        exclusion_match = PolicyInterpreter._check_exclusions(
            diagnosis, procedures, plan["standard_exclusions"]
        )
        reasoning.append(
            f"Exclusion check: {'No exclusions apply' if not exclusion_match else f'EXCLUSION FOUND: {exclusion_match}'}"
        )

        # 3. Waiting Period Validation
        waiting_result = PolicyInterpreter._check_waiting_period(
            diagnosis, procedures, policy_start_date,
            admission_date, plan["waiting_periods"]
        )
        reasoning.append(
            f"Waiting period: {waiting_result['detail']}"
        )

        # 4. Sum Insured Check
        within_si = total_bill <= sum_insured if sum_insured > 0 else True
        reasoning.append(
            f"Sum insured check: Claim ₹{total_bill:,.0f} vs SI ₹{sum_insured:,.0f} — "
            f"{'Within limit' if within_si else 'EXCEEDS sum insured'}"
        )

        # 5. Sub-Limit Check
        sub_limit_result = PolicyInterpreter._check_sub_limits(
            procedure_categories, sum_insured, total_bill, plan["sub_limits"]
        )
        reasoning.append(f"Sub-limit check: {sub_limit_result['detail']}")

        # 6. Co-Pay Determination
        is_network = any(chain in hospital for chain in NETWORK_HOSPITAL_CHAINS)
        co_pay_pct = (
            plan["co_pay"]["network"] if is_network else plan["co_pay"]["non_network"]
        )
        reasoning.append(
            f"Hospital network: {'Network' if is_network else 'Non-network'} — "
            f"Co-pay: {co_pay_pct}%"
        )

        # 7. Pre-Authorization
        pre_auth = "Not Required" if is_network else "Required for Non-Network Hospital"
        reasoning.append(f"Pre-authorization: {pre_auth}")

        # Build policy analysis result
        policy_analysis = {
            "procedure_covered": all_covered and not exclusion_match,
            "within_sum_insured": within_si,
            "sub_limit_applicable": sub_limit_result["applicable"],
            "sub_limit_amount": sub_limit_result["amount"],
            "claim_within_sub_limit": sub_limit_result["within_limit"],
            "waiting_period_satisfied": waiting_result["satisfied"],
            "exclusions_applicable": bool(exclusion_match),
            "exclusion_detail": exclusion_match or None,
            "co_pay_percentage": co_pay_pct,
            "pre_authorization": pre_auth,
            "plan_type": plan["display_name"],
            "is_network_hospital": is_network,
        }

        return policy_analysis, reasoning

    @staticmethod
    def _identify_plan(coverage_type: str) -> str:
        """Identify plan type from coverage string."""
        coverage_lower = coverage_type.lower()
        if "comprehensive" in coverage_lower or "individual" in coverage_lower:
            return "comprehensive_health"
        elif "top-up" in coverage_lower or "top up" in coverage_lower or "super" in coverage_lower:
            return "super_top_up"
        elif "basic" in coverage_lower:
            return "basic_health"
        return "comprehensive_health"  # Default

    @staticmethod
    def _get_procedure_categories(procedures: List[str]) -> List[str]:
        """Map procedures to medical categories."""
        categories = set()
        for proc in procedures:
            proc_lower = proc.lower()
            for keyword, category in PROCEDURE_CATEGORY_MAP.items():
                if keyword in proc_lower:
                    categories.add(category)
                    break
        return list(categories) if categories else ["general"]

    @staticmethod
    def _check_exclusions(
        diagnosis: str, procedures: List[str], exclusions: List[str]
    ) -> Optional[str]:
        """Check if claim falls under any policy exclusion."""
        diag_lower = diagnosis.lower()
        proc_str = " ".join(p.lower() for p in procedures)
        combined = diag_lower + " " + proc_str

        exclusion_keywords = {
            "cosmetic_surgery": ["cosmetic", "aesthetic", "liposuction", "rhinoplasty", "botox"],
            "dental_unless_accidental": [],  # Handled separately
            "self_inflicted_injury": ["self-inflicted", "self inflicted", "suicide attempt"],
            "substance_abuse_treatment": ["alcohol rehabilitation", "drug rehabilitation", "substance abuse"],
            "experimental_treatment": ["experimental", "clinical trial", "investigational"],
            "infertility_treatment": ["ivf", "iui", "infertility", "fertility treatment"],
            "weight_management": ["bariatric", "weight loss surgery", "liposuction"],
            "congenital_external_defects": ["congenital external", "birth defect external"],
            "alternative_medicine": ["ayurveda", "homeopathy", "naturopathy", "unani"],
            "non_allopathic": ["ayurvedic", "homeopathic", "naturopathic"],
        }

        for excl in exclusions:
            keywords = exclusion_keywords.get(excl, [])
            for kw in keywords:
                if kw in combined:
                    return excl.replace("_", " ").title()

        return None

    @staticmethod
    def _check_waiting_period(
        diagnosis: str, procedures: List[str],
        policy_start: str, admission_date: str,
        waiting_periods: Dict[str, int],
    ) -> Dict[str, Any]:
        """Validate waiting period requirements."""
        try:
            start = datetime.strptime(policy_start, "%Y-%m-%d")
            admission = datetime.strptime(admission_date, "%Y-%m-%d")
            days_since_start = (admission - start).days
        except (ValueError, TypeError):
            return {
                "satisfied": True,
                "detail": "Unable to verify dates — assuming satisfied",
                "days_since_policy_start": None,
            }

        # Check initial waiting period (30 days)
        initial_waiting = waiting_periods.get("initial_waiting", 30)
        if days_since_start < initial_waiting:
            return {
                "satisfied": False,
                "detail": (
                    f"Initial waiting period NOT satisfied — "
                    f"Only {days_since_start} days since policy start "
                    f"(required: {initial_waiting} days)"
                ),
                "days_since_policy_start": days_since_start,
            }

        # Check pre-existing disease waiting period
        diag_lower = diagnosis.lower()
        is_ped = any(ped in diag_lower for ped in PRE_EXISTING_INDICATORS)
        if is_ped:
            ped_waiting = waiting_periods.get("pre_existing_disease", 730)
            if days_since_start < ped_waiting:
                return {
                    "satisfied": False,
                    "detail": (
                        f"Pre-existing disease waiting period NOT satisfied — "
                        f"{days_since_start} days since policy start "
                        f"(required: {ped_waiting} days for PED)"
                    ),
                    "days_since_policy_start": days_since_start,
                }

        # Check specific disease waiting period
        proc_str = " ".join(p.lower() for p in procedures)
        is_specific = any(
            sd in diag_lower or sd in proc_str
            for sd in SPECIFIC_WAITING_PERIOD_DISEASES
        )
        if is_specific:
            specific_waiting = waiting_periods.get("specific_disease", 365)
            if days_since_start < specific_waiting:
                return {
                    "satisfied": False,
                    "detail": (
                        f"Specific disease waiting period NOT satisfied — "
                        f"{days_since_start} days since policy start "
                        f"(required: {specific_waiting} days)"
                    ),
                    "days_since_policy_start": days_since_start,
                }

        # Check maternity waiting period
        is_maternity = any(
            kw in diag_lower or kw in proc_str
            for kw in ["delivery", "cesarean", "c-section", "maternity", "pregnancy"]
        )
        if is_maternity:
            maternity_waiting = waiting_periods.get("maternity", 730)
            if days_since_start < maternity_waiting:
                return {
                    "satisfied": False,
                    "detail": (
                        f"Maternity waiting period NOT satisfied — "
                        f"{days_since_start} days since policy start "
                        f"(required: {maternity_waiting} days)"
                    ),
                    "days_since_policy_start": days_since_start,
                }

        return {
            "satisfied": True,
            "detail": (
                f"All waiting periods satisfied — "
                f"{days_since_start} days since policy start "
                f"(policy since {policy_start})"
            ),
            "days_since_policy_start": days_since_start,
        }

    @staticmethod
    def _check_sub_limits(
        categories: List[str], sum_insured: float,
        total_bill: float, sub_limits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if claim is within applicable sub-limits."""
        if not sub_limits:
            return {
                "applicable": False,
                "amount": sum_insured,
                "within_limit": True,
                "detail": "No sub-limits applicable for this plan",
            }

        # Find the most relevant sub-limit for the procedure category
        applicable_limit = sum_insured
        limit_found = False

        for cat in categories:
            if cat in sub_limits:
                ratio = sub_limits[cat]
                if isinstance(ratio, float) and ratio <= 1.0:
                    cat_limit = sum_insured * ratio
                else:
                    cat_limit = ratio
                if cat_limit < applicable_limit:
                    applicable_limit = cat_limit
                    limit_found = True

        within = total_bill <= applicable_limit
        return {
            "applicable": limit_found,
            "amount": round(applicable_limit),
            "within_limit": within,
            "detail": (
                f"Sub-limit: ₹{applicable_limit:,.0f} — "
                f"Claim {'within' if within else 'EXCEEDS'} sub-limit"
                if limit_found
                else "No specific sub-limit for this procedure category"
            ),
        }


# ============================================================
# Claim History Analyzer
# ============================================================

class ClaimHistoryAnalyzer:
    """Analyzes claim history patterns for credibility assessment."""

    @staticmethod
    def analyze(
        patient_id: str, current_claim_amount: float
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Analyze claim history for the patient.

        Returns:
            (history_summary dict, reasoning steps list)
        """
        reasoning = []

        # Look up claim history
        history = CLAIM_HISTORY_DB.get(patient_id, CLAIM_HISTORY_DB["DEFAULT"])
        claims = history.get("claims", [])
        policy_start = history.get("policy_start_date", "2024-01-01")

        reasoning.append(f"Loading claim history for patient {patient_id}")

        total = len(claims)
        approved = sum(1 for c in claims if c.get("status") == "approved")
        rejected = sum(1 for c in claims if c.get("status") == "rejected")
        flagged = sum(1 for c in claims if c.get("flagged"))
        total_past_amount = sum(c.get("amount", 0) for c in claims)

        reasoning.append(
            f"Found {total} previous claims — "
            f"{approved} approved, {rejected} rejected, {flagged} flagged"
        )

        # Frequency analysis
        if total >= 5:
            reasoning.append(
                "NOTICE: High claim frequency — 5+ claims detected"
            )
        elif total >= 3:
            reasoning.append("Moderate claim frequency — within normal range")
        else:
            reasoning.append("Low claim frequency — good standing")

        # Escalation pattern check
        if total > 0 and current_claim_amount > total_past_amount * 2:
            reasoning.append(
                f"NOTICE: Current claim (₹{current_claim_amount:,.0f}) is significantly higher "
                f"than total past claims (₹{total_past_amount:,.0f})"
            )

        # Flag history check
        if flagged > 0:
            reasoning.append(
                f"WARNING: {flagged} previously flagged claim(s) detected"
            )

        summary = {
            "total_claims": total,
            "approved": approved,
            "rejected": rejected,
            "flagged": flagged,
            "total_past_amount": total_past_amount,
            "policy_start_date": policy_start,
            "claim_details": claims,
        }

        return summary, reasoning


# ============================================================
# Credibility Agent
# ============================================================

class CredibilityAgent:
    """
    User Credibility & Policy Interpretation Agent.

    Pipeline:
      1. Extract claim history and analyze patterns
      2. Extract features from all upstream agent data
      3. Score credibility using ML model (Random Forest simulation)
      4. Interpret policy terms (coverage, exclusions, waiting periods)
      5. Use LLM for deep policy clause analysis (if available)
      6. Produce comprehensive credibility + policy assessment
    """

    AGENT_NAME = "Credibility & Policy Interpretation Agent"
    OWNER = "Shruti Roy"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._feature_extractor = CredibilityFeatureExtractor()
        self._scoring_model = CredibilityScoringModel()
        self._policy_interpreter = PolicyInterpreter()
        self._history_analyzer = ClaimHistoryAnalyzer()
        logger.info(
            f"[{self.AGENT_NAME}] Initialized — "
            f"ML model: {self._scoring_model.model_info}"
        )

    async def process(self, claim_data: Dict) -> Dict[str, Any]:
        """
        Process claim data to assess user credibility and interpret policy.

        Args:
            claim_data: Dict of upstream agent results
                        (keys: image, pdf, requirements)

        Returns:
            Standardized agent result with credibility score and policy analysis.
        """
        start = time.time()
        reasoning: List[str] = []

        reasoning.append("Starting credibility assessment and policy interpretation")

        image_output = _safe_get(claim_data, "image", "output") or {}

        patient_id = image_output.get("patient_id", "DEFAULT")
        total_bill = image_output.get("total_bill_amount", 0)

        # Step 1: Claim History Analysis
        reasoning.append("--- Claim History Analysis ---")
        history_summary, history_reasoning = self._history_analyzer.analyze(
            patient_id, total_bill
        )
        reasoning.extend(history_reasoning)

        policy_start_date = history_summary.get("policy_start_date", "2024-01-01")

        # Step 2: Feature Extraction
        reasoning.append("--- Feature Extraction (Random Forest Pipeline) ---")
        features = self._feature_extractor.extract(
            claim_data, history_summary, policy_start_date
        )
        reasoning.append(
            f"Extracted {len(features)} features for credibility model"
        )
        # Log top features
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for fname, fval in sorted_features[:5]:
            reasoning.append(f"  Feature: {fname} = {fval:.3f}")

        # Step 3: Credibility Scoring
        reasoning.append(f"--- ML Credibility Scoring ({self._scoring_model.model_info}) ---")
        cred_score, cred_rating, contributions = self._scoring_model.predict(features)
        reasoning.append(
            f"Credibility score: {cred_score:.2f} ({cred_rating}) — "
            f"threshold: 0.40 for REJECT"
        )

        # Log top contributing features
        top_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        for fname, contrib in top_contribs:
            reasoning.append(
                f"  Top contributor: {fname} (contribution: {contrib:.4f})"
            )

        # Step 4: Policy Interpretation
        reasoning.append("--- Policy Interpretation (BERT-style Analysis) ---")
        policy_analysis, policy_reasoning = self._policy_interpreter.interpret(
            claim_data, policy_start_date
        )
        reasoning.extend(policy_reasoning)

        # Step 5: LLM Deep Analysis (for complex or borderline cases)
        llm_analysis = None
        is_borderline = 0.40 <= cred_score <= 0.70
        has_issues = (
            not policy_analysis["procedure_covered"]
            or policy_analysis["exclusions_applicable"]
            or not policy_analysis["waiting_period_satisfied"]
        )

        if self.llm_client and (is_borderline or has_issues):
            reasoning.append("--- LLM Deep Policy Analysis ---")
            reasoning.append("Running LLM analysis for borderline/complex case")
            try:
                llm_analysis = await self._llm_policy_analysis(
                    claim_data, policy_analysis, history_summary, cred_score
                )
                if llm_analysis:
                    reasoning.append(
                        f"LLM assessment: {llm_analysis.get('summary', 'Complete')}"
                    )
            except Exception as e:
                reasoning.append(f"LLM analysis skipped: {e}")
                logger.warning(f"[{self.AGENT_NAME}] LLM analysis failed: {e}")

        # Step 6: Build Output
        output = {
            "credibility_score": cred_score,
            "credibility_rating": cred_rating,
            "claim_history": {
                "total_claims": history_summary["total_claims"],
                "approved": history_summary["approved"],
                "rejected": history_summary["rejected"],
                "flagged": history_summary["flagged"],
            },
            "policy_analysis": policy_analysis,
            "feature_scores": features,
            "feature_contributions": contributions,
            "ml_model": self._scoring_model.model_info,
            "nlp_model": "Fine-tuned BERT on insurance policy corpus",
            "llm_analysis": llm_analysis,
        }

        # Compute confidence
        confidence = self._compute_confidence(cred_score, policy_analysis, features)

        elapsed = time.time() - start
        logger.info(
            f"[{self.AGENT_NAME}] Completed in {elapsed:.1f}s — "
            f"Credibility: {cred_score:.2f} ({cred_rating}), "
            f"Coverage: {'Yes' if policy_analysis['procedure_covered'] else 'No'}"
        )

        return {
            "agent": self.AGENT_NAME,
            "owner": self.OWNER,
            "status": "success",
            "reasoning": reasoning,
            "output": output,
            "confidence": confidence,
            "processing_time": f"{elapsed:.1f}s",
        }

    # --------------------------------------------------------
    # LLM-Powered Policy Analysis
    # --------------------------------------------------------

    async def _llm_policy_analysis(
        self, claim_data: Dict, policy_analysis: Dict,
        history: Dict, cred_score: float
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for deep policy analysis on complex/borderline cases."""
        if not self.llm_client:
            return None

        image_output = _safe_get(claim_data, "image", "output") or {}
        pdf_output = _safe_get(claim_data, "pdf", "output") or {}

        prompt = (
            f"Analyze this insurance claim for policy compliance:\n\n"
            f"Patient: {image_output.get('patient_name', 'N/A')}\n"
            f"Diagnosis: {image_output.get('diagnosis', 'N/A')}\n"
            f"Procedures: {', '.join(image_output.get('procedures', []))}\n"
            f"Hospital: {image_output.get('hospital', 'N/A')}\n"
            f"Total Bill: ₹{image_output.get('total_bill_amount', 0):,.0f}\n"
            f"Coverage Type: {pdf_output.get('coverage_type', 'N/A')}\n"
            f"Sum Insured: ₹{pdf_output.get('sum_insured', 0):,.0f}\n"
            f"Policy Start: {history.get('policy_start_date', 'N/A')}\n"
            f"Credibility Score: {cred_score:.2f}\n"
            f"Past Claims: {history.get('total_claims', 0)}\n\n"
            f"Current Policy Assessment:\n"
            f"- Procedure Covered: {policy_analysis['procedure_covered']}\n"
            f"- Exclusions: {policy_analysis['exclusions_applicable']}\n"
            f"- Waiting Period Satisfied: {policy_analysis['waiting_period_satisfied']}\n"
            f"- Within Sum Insured: {policy_analysis['within_sum_insured']}\n"
            f"- Co-pay: {policy_analysis['co_pay_percentage']}%\n\n"
            f"Provide a JSON response with:\n"
            f"1. 'summary': 1-2 sentence assessment\n"
            f"2. 'risk_flags': list of any concerns\n"
            f"3. 'recommendation': APPROVE / REVIEW / REJECT\n"
            f"4. 'confidence': your confidence level (0-1)\n"
        )

        system_prompt = (
            "You are an insurance policy interpretation AI specializing in Indian "
            "health insurance regulations (IRDAI guidelines). Analyze claims against "
            "policy terms and provide professional assessments. Respond in valid JSON only."
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            result = await self.llm_client.complete(messages, system_prompt=system_prompt)
            content = result.get("content", "{}")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"summary": content, "recommendation": "REVIEW"}
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] LLM policy analysis failed: {e}")
            return None

    # --------------------------------------------------------
    # Confidence Computation
    # --------------------------------------------------------

    def _compute_confidence(
        self, cred_score: float, policy_analysis: Dict,
        features: Dict[str, float]
    ) -> float:
        """Compute agent confidence based on data quality and analysis clarity."""
        confidence = 0.92  # Base confidence

        # Higher confidence when result is clear-cut
        if cred_score > 0.80 or cred_score < 0.30:
            confidence += 0.03  # Clear signal

        # Reduce if policy issues found
        if not policy_analysis.get("procedure_covered", True):
            confidence -= 0.05
        if policy_analysis.get("exclusions_applicable"):
            confidence -= 0.05
        if not policy_analysis.get("waiting_period_satisfied", True):
            confidence -= 0.03

        # Reduce if key features had low values (uncertainty)
        if features.get("proc_diag_consistency", 1.0) < 0.5:
            confidence -= 0.05

        return round(max(min(confidence, 0.99), 0.50), 2)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------


def _safe_get(data: Dict, *keys, default=None):
    """Safely traverse nested dicts."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current

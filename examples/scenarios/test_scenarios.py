#!/usr/bin/env python3
"""
Test script to run all example scenarios through the Insurance Claim AI system.
Usage: python test_scenarios.py
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
SCENARIOS_DIR = Path(__file__).parent

# Scenario definitions
SCENARIOS = {
    "valid_claim": {
        "name": "Valid Claim - Should be Approved",
        "file": "valid_claim/claim_document.txt",
        "expected_decision": "APPROVED",
        "expected_confidence": 0.90,
    },
    "missing_documents": {
        "name": "Missing Documents - Should be Rejected",
        "file": "missing_documents/claim_document.txt",
        "expected_decision": "REJECTED",
        "expected_missing_fields": ["discharge_date"],
    },
    "duplicate_claim": {
        "name": "Duplicate Claim - Should be Flagged",
        "file": "duplicate_claim/claim_document_2.txt",
        "expected_duplicate": True,
        "expected_similarity": 0.85,
    },
    "fraud_suspicious": {
        "name": "Fraud Suspicious - Should be Held",
        "file": "fraud_suspicious/claim_document.txt",
        "expected_decision": "HOLD",
        "expected_fraud_score": 0.70,
    },
    "high_value_claim": {
        "name": "High Value Claim - Should be Approved with Review",
        "file": "high_value_claim/claim_document.txt",
        "expected_decision": "APPROVED",
        "expected_review_required": True,
    },
    "pre_existing_condition": {
        "name": "Pre-Existing Condition - Should be Approved",
        "file": "pre_existing_condition/claim_document.txt",
        "expected_decision": "APPROVED",
        "expected_ped_satisfied": True,
    },
}


def load_claim_document(file_path: str) -> str:
    """Load claim document from file"""
    full_path = SCENARIOS_DIR / file_path
    with open(full_path, 'r') as f:
        return f.read()


def process_claim(claim_text: str) -> Dict[str, Any]:
    """Submit claim to API for processing"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/claims/process",
            json={"claim_text": claim_text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def validate_scenario(scenario_id: str, scenario_config: Dict, result: Dict) -> Dict[str, Any]:
    """Validate scenario results against expected outcomes"""
    validation = {
        "scenario": scenario_id,
        "name": scenario_config["name"],
        "passed": True,
        "checks": []
    }
    
    if "error" in result:
        validation["passed"] = False
        validation["checks"].append({
            "check": "API Call",
            "status": "FAILED",
            "message": result["error"]
        })
        return validation
    
    # Check expected decision
    if "expected_decision" in scenario_config:
        actual_decision = result.get("final_decision", {}).get("decision", "UNKNOWN")
        expected = scenario_config["expected_decision"]
        passed = actual_decision == expected
        validation["checks"].append({
            "check": "Final Decision",
            "status": "PASSED" if passed else "FAILED",
            "expected": expected,
            "actual": actual_decision
        })
        if not passed:
            validation["passed"] = False
    
    # Check missing fields
    if "expected_missing_fields" in scenario_config:
        actual_missing = result.get("agents", {}).get("requirements_agent", {}).get("output", {}).get("missing_fields", [])
        expected = scenario_config["expected_missing_fields"]
        passed = set(actual_missing) == set(expected)
        validation["checks"].append({
            "check": "Missing Fields",
            "status": "PASSED" if passed else "FAILED",
            "expected": expected,
            "actual": actual_missing
        })
        if not passed:
            validation["passed"] = False
    
    # Check duplicate detection
    if "expected_duplicate" in scenario_config:
        actual_duplicate = result.get("agents", {}).get("requirements_agent", {}).get("output", {}).get("duplicate_detection", {}).get("is_duplicate", False)
        expected = scenario_config["expected_duplicate"]
        passed = actual_duplicate == expected
        validation["checks"].append({
            "check": "Duplicate Detection",
            "status": "PASSED" if passed else "FAILED",
            "expected": expected,
            "actual": actual_duplicate
        })
        if not passed:
            validation["passed"] = False
    
    return validation


def print_results(validations: list):
    """Print test results in a formatted way"""
    print("\n" + "="*80)
    print("INSURANCE CLAIM AI - SCENARIO TEST RESULTS")
    print("="*80 + "\n")
    
    total = len(validations)
    passed = sum(1 for v in validations if v["passed"])
    
    for validation in validations:
        status_icon = "✅" if validation["passed"] else "❌"
        print(f"{status_icon} {validation['name']}")
        
        for check in validation["checks"]:
            check_icon = "  ✓" if check["status"] == "PASSED" else "  ✗"
            print(f"{check_icon} {check['check']}: {check['status']}")
            if check["status"] == "FAILED":
                print(f"    Expected: {check.get('expected', 'N/A')}")
                print(f"    Actual: {check.get('actual', 'N/A')}")
        print()
    
    print("="*80)
    print(f"SUMMARY: {passed}/{total} scenarios passed")
    print("="*80 + "\n")
    
    return passed == total


def main():
    """Run all scenario tests"""
    print("Starting Insurance Claim AI Scenario Tests...")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Scenarios Directory: {SCENARIOS_DIR}\n")
    
    validations = []
    
    for scenario_id, scenario_config in SCENARIOS.items():
        print(f"Testing: {scenario_config['name']}...")
        
        # Load claim document
        claim_text = load_claim_document(scenario_config["file"])
        
        # Process claim
        result = process_claim(claim_text)
        
        # Validate results
        validation = validate_scenario(scenario_id, scenario_config, result)
        validations.append(validation)
    
    # Print results
    all_passed = print_results(validations)
    
    # Exit with appropriate code
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

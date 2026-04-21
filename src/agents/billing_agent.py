import asyncio
import time
import logging
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("insurance_claim_ai.agents")

class BillingAgent:
    AGENT_NAME = "Billing & Processing Agent"
    OWNER = "SSpandana"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        logger.info(f"[{self.AGENT_NAME}] Initialized")

    def _parse_amount(self, value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        cleaned = re.sub(r"[^\d.]", "", str(value))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def _collect_inputs(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        requirements_output = claim_data.get("requirements", {}).get("output", {})
        extracted_requirements = requirements_output.get("extracted_requirements", {}) or {}

        image_text = claim_data.get("image", {}).get("output", {}).get("extracted_text", "") or ""
        pdf_text = claim_data.get("pdf", {}).get("output", {}).get("extracted_text", "") or ""

        combined_text = f"{image_text}\n{pdf_text}".strip()

        return {
        "extracted_requirements": extracted_requirements,
        "image_text": image_text,
        "pdf_text": pdf_text,
        "combined_text": combined_text,
        }

    def _get_category_keywords(self) -> Dict[str, List[str]]:
        return {
        "room_charges": [
        "room rent", "room charges", "bed charges", "icu rent",
        "nursing", "routine service", "ward charges", "twin sharing"
        ],
        "procedure_charges": [
        "procedure", "surgery", "operation", "operation theatre",
        "ot charges", "surgical", "appendectomy", "angioplasty", "stent"
        ],
        "medication": [
        "medicine", "medicines", "drug", "pharmacy",
        "tablet", "capsule", "injection", "antibiotic"
        ],
        "investigations": [
        "lab", "laboratory", "test", "scan", "xray", "x-ray",
        "mri", "ct", "ecg", "echo", "ultrasound", "investigation"
        ],
        "consumables": [
        "consumable", "gloves", "syringe", "catheter", "dressing", "sutures"
        ],
        "doctor_fees": [
        "doctor fee", "consultation", "surgeon fee", "assistant surgeon",
        "anesthetist", "doctor visit", "consultant", "professional fee",
        "visit charges"
        ],
        "non_payable": [
        "registration", "file", "welcome kit", "kit charge","convenience kit",
        "attender", "meal","food","service charge", "admission charge"
        ],
        }

    def _match_category(self, text: str) -> str:
        lower = text.lower()
        if any(x in lower for x in [
            "registration",
            "file",
            "kit",
            "convenience",
            "attender",
            "meal",
            "food",
            "service charge"
        ]):
            return "non-payable"
            
        category_keywords = self._get_category_keywords()

        for category, keywords in category_keywords.items():
            if category =="non_payable":
                continue
            if any(keyword in lower for keyword in keywords):
                return category
        return "miscellaneous"

    def _extract_line_items(self, text: str) -> List[Tuple[str, float, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        line_items: List[Tuple[str, float, str]] = []

        skip_keywords = [
        "grand total",
        "total amount",
        "claim amount submitted",
        "net payable",
        "subtotal",
        "bill total",
        ]

        seen_entries = set()

        print("[BILLING][DEBUG] raw line count:", len(lines))
        for idx, line in enumerate(lines[:50], 1):
            print(f"[BILLING][DEBUG] line {idx}: {line}")

        for i, line in enumerate(lines):
            lower = line.lower()

            if any(keyword in lower for keyword in skip_keywords):
                print("[BILLING][SKIP] summary line skipped:", line)
                continue

            # Try current line alone, then merge with next lines to handle multiline PDF extraction
            candidates = [line]

            if i + 1 < len(lines):
                candidates.append(f"{line} {lines[i + 1]}")

            if i + 2 < len(lines):
                candidates.append(f"{line} {lines[i + 1]} {lines[i + 2]}")

            matched = False

            for candidate in candidates:
                candidate_lower = candidate.lower()

                if any(keyword in candidate_lower for keyword in skip_keywords):
                    continue

                category = self._match_category(candidate)
                if category == "miscellaneous":
                    continue

                amount_matches = re.findall(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)", candidate_lower)
                if not amount_matches:
                    continue

                # take the largest amount in the candidate since multiline rows may contain qty + unit rate + total
                parsed_amounts = [self._parse_amount(x) for x in amount_matches]
                parsed_amounts = [x for x in parsed_amounts if x > 0]

                if not parsed_amounts:
                    continue

                amount = max(parsed_amounts)

                if amount < 10:
                    print("[BILLING][SKIP] tiny amount skipped:", candidate)
                    continue

                dedupe_key = (category, amount, re.sub(r"\s+", " ", candidate_lower).strip())
                if dedupe_key in seen_entries:
                    continue
                seen_entries.add(dedupe_key)
                print(f"[BILLING][CATEGORY matched]: {category} | amount: {amount} |text: {candidate}" )
                line_items.append((category, amount, candidate))
                matched = True
                break

        # Fallback: if single line has amount but no category, keep it as miscellaneous
            if not matched:
                amount_matches = re.findall(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)", lower)
                if amount_matches:
                    parsed_amounts = [self._parse_amount(x) for x in amount_matches]
                    parsed_amounts = [x for x in parsed_amounts if x >= 10]
                    if parsed_amounts:
                        amount = max(parsed_amounts)
                        dedupe_key = ("miscellaneous", amount, re.sub(r"\s+", " ", lower).strip())
                        if dedupe_key not in seen_entries:
                            seen_entries.add(dedupe_key)
                            line_items.append(("miscellaneous", amount, line))

        return line_items

    def _build_breakdown(self, line_items: List[Tuple[str, float, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {}

        for category, amount, raw_line in line_items:
            if category not in breakdown:
                breakdown[category] = {
                "claimed": 0.0,
                "approved": 0.0,
                "status": "approved",
                "reason": "Within expected range",
                "lines": []
                }

            breakdown[category]["claimed"] += amount
            breakdown[category]["approved"] += amount
            breakdown[category]["lines"].append(raw_line)

        total_claim_amount = self._parse_amount(context.get("total_claim_amount"))

        for category, entry in breakdown.items():
            claimed = entry["claimed"]

            if category == "non_payable":
                entry["approved"] = 0.0
                entry["status"] = "rejected"
                entry["reason"] = "Non-medical or non-payable charge"

            elif category == "room_charges" and claimed > 50000:
                entry["approved"] = min(claimed, 50000)
                entry["status"] = "partial"
                entry["reason"] = "Room charges exceed default cap"

            elif category == "doctor_fees" and total_claim_amount and claimed > (0.15 * total_claim_amount):
                entry["approved"] = 0.15 * total_claim_amount
                entry["status"] = "partial"
                entry["reason"] = "Doctor fees exceed 15% threshold"

            elif category == "consumables" and total_claim_amount and claimed > (0.20 * total_claim_amount):
                entry["approved"] = 0.20 * total_claim_amount
                entry["status"] = "partial"
                entry["reason"] = "Consumables exceed 20% threshold"
            elif category == "miscellaneous" and claimed < 20000:
                entry["approved"] = 0.0
                entry["status"] = "rejected"
                entry["reason"] = "Small miscellaneous charge treated as non-payable"

            entry["approved"] = min(entry["approved"], entry["claimed"])

        return breakdown

    def _score_anomalies(self, breakdown: Dict[str, Any], declared_total: float) -> float:
        score = 0.0

        total_claimed = sum(item["claimed"] for item in breakdown.values())
        total_approved = sum(item["approved"] for item in breakdown.values())

        if declared_total > 0:
            mismatch_ratio = abs(total_claimed - declared_total) / max(declared_total, 1)
            if mismatch_ratio > 0.20:
                score += 0.15

        partial_count = sum(1 for item in breakdown.values() if item["status"] == "partial")
        score += min(partial_count * 0.10, 0.30)

        rejected_count = sum(1 for item in breakdown.values() if item["status"] == "rejected")
        score += min(rejected_count * 0.08, 0.20)

        if total_claimed > 0 and (total_claimed - total_approved) / total_claimed > 0.25:
            score += 0.20

        return min(score, 1.0)

    async def process(self, claim_data: Dict) -> Dict[str, Any]:
        start = time.time()

        print("\n" + "=" * 80)
        print("[BILLING] START")
        print("=" * 80)

        inputs = self._collect_inputs(claim_data)
        extracted_requirements = inputs["extracted_requirements"]
        combined_text = inputs["combined_text"]

        print("[BILLING][INPUT] claim_data keys:", list(claim_data.keys()))
        print("[BILLING][INPUT] extracted_requirements:", extracted_requirements)
        print("[BILLING][INPUT] image_text_length:", len(inputs["image_text"]))
        print("[BILLING][INPUT] pdf_text_length:", len(inputs["pdf_text"]))
        print("[BILLING][INPUT] combined_text_length:", len(combined_text))
        print("[BILLING][INPUT] combined_text_preview:")
        print(combined_text[:1200])
        print("[BILLING][DEBUG] combined_text repr:")
        print(repr(combined_text[:1200]))

        line_items = self._extract_line_items(combined_text)
        print("[BILLING][PARSE] extracted_line_items_count:", len(line_items))
        for item in line_items:
            print("[BILLING][PARSE] item:", item)

        context = {
        "total_claim_amount": extracted_requirements.get("total_claim_amount"),
        "diagnosis": extracted_requirements.get("diagnosis"),
        "hospital_name": extracted_requirements.get("hospital_name"),
        "admission_date": extracted_requirements.get("admission_date"),
        "discharge_date": extracted_requirements.get("discharge_date"),
        }

        breakdown = self._build_breakdown(line_items, context)

        declared_total = self._parse_amount(extracted_requirements.get("total_claim_amount"))
        print("[BILLING][DEBUG] declared_total parsed:", declared_total)
        parsed_total = sum(item["claimed"] for item in breakdown.values())
        approved_total_before_clamp = sum(item["approved"] for item in breakdown.values())

        effective_claimed_total = declared_total if declared_total > 0 else parsed_total
        approved_total = min(approved_total_before_clamp, effective_claimed_total)

        if parsed_total > 0:
            approved_total = min(approved_total, parsed_total)

        deductions = max(effective_claimed_total - approved_total, 0.0)
        anomaly_score = self._score_anomalies(breakdown, declared_total)

        print("[BILLING][TOTALS] declared_total:", declared_total)
        print("[BILLING][TOTALS] parsed_total:", parsed_total)
        print("[BILLING][TOTALS] approved_total_before_clamp:", approved_total_before_clamp)
        print("[BILLING][TOTALS] effective_claimed_total:", effective_claimed_total)
        print("[BILLING][TOTALS] approved_total_after_clamp:", approved_total)
        print("[BILLING][TOTALS] deductions:", deductions)

        reasoning = [
        "Collected upstream document text from image/PDF agents",
        "Extracted billing line items using multiline-aware parsing",
        "Grouped charges into billing categories",
        "Applied rule-based approval thresholds by category",
        f"Parsed total claimed: {parsed_total:.2f}",
        f"Approved total: {approved_total:.2f}",
        f"Anomaly score computed as {anomaly_score:.2f}",
        ]
        if deductions>0:
            reasoning.append(f"Detected non-payable charges totaling ₹{deductions:.2f}")
        if anomaly_score>0:
            reasoning.append(f"Billing anomaly score assessed at ₹{anomaly_score:.2f}")
        output = {
        "total_claimed": effective_claimed_total,
        "total_approved": approved_total,
        "breakdown": breakdown,
        "anomaly_score": anomaly_score,
        "pricing_benchmark": "Rule-based benchmark v3",
        "deductions": deductions,
        }
        print("[BILLING][OUTPUT] breakdown categories:", list(breakdown.keys()))
        print("[BILLING][OUTPUT] breakdown:", breakdown)
        print("[BILLING][OUTPUT] total_claimed:", output["total_claimed"])
        print("[BILLING][OUTPUT] total_approved:", output["total_approved"])
        print("[BILLING][OUTPUT] deductions:", output["deductions"])
        print("[BILLING][OUTPUT] anomaly_score:", output["anomaly_score"])
        print("[BILLING] END")
        print("=" * 80 + "\n")

        await asyncio.sleep(0.1)

        elapsed = time.time() - start
        return {
        "agent": self.AGENT_NAME,
        "owner": self.OWNER,
        "status": "success",
        "reasoning": reasoning,
        "output": output,
        "confidence": max(0.55, 1.0 - anomaly_score / 2),
        "processing_time": f"{elapsed:.1f}s",
        }
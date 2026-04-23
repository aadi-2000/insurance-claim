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
                "room rent",
                "room charges",
                "bed charges",
                "icu rent",
                "nursing",
                "routine service",
                "ward charges",
                "twin sharing",
                "single deluxe",
                "deluxe room",
                "monitoring charges",
            ],
            "procedure_charges": [
                "procedure",
                "surgery",
                "operation",
                "operation theatre",
                "ot charges",
                "surgical",
                "appendectomy",
                "angioplasty",
                "stent",
            ],
            "medication": [
                "medicine",
                "medicines",
                "drug",
                "pharmacy",
                "tablet",
                "capsule",
                "injection",
                "antibiotic",
            ],
            "investigations": [
                "lab",
                "laboratory",
                "test",
                "scan",
                "xray",
                "x-ray",
                "mri",
                "ct",
                "ecg",
                "echo",
                "ultrasound",
                "investigation",
                "blood panel",
            ],
            "consumables": [
                "consumable",
                "gloves",
                "syringe",
                "catheter",
                "dressing",
                "sutures",
            ],
            "doctor_fees": [
                "doctor fee",
                "consultation",
                "surgeon fee",
                "assistant surgeon",
                "anesthetist",
                "anaesthetist",
                "doctor visit",
                "consultant",
                "professional fee",
                "visit charges",
                "specialist visit",
            ],
            "non_payable": [
                "registration",
                "file",
                "file handling",
                "welcome kit",
                "kit charge",
                "convenience kit",
                "patient kit",
                "attender",
                "meal",
                "food",
                "service charge",
                "admission charge",
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
            "service charge",
            "admission charge",
        ]):
            return "non_payable"

        category_keywords = self._get_category_keywords()
        for category, keywords in category_keywords.items():
            if category == "non_payable":
                continue
            if any(keyword in lower for keyword in keywords):
                return category

        return "miscellaneous"

    def _extract_line_items(self, text: str) -> List[Tuple[str, float, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        line_items: List[Tuple[str, float, str]] = []

        print("[BILLING][DEBUG] raw line count:", len(lines))
        for idx, line in enumerate(lines[:120], 1):
            print(f"[BILLING][DEBUG] line {idx}: {line}")

        seen_entries = set()

        # Only parse the itemized bill section between Page 2 table start and Page 3 summary
        start_idx = None
        end_idx = None

        for idx, line in enumerate(lines):
            if line.lower() == "description" and idx + 3 < len(lines):
                if lines[idx + 1].lower() == "qty" and lines[idx + 2].lower() == "rate" and lines[idx + 3].lower() == "amount":
                    start_idx = idx + 4
                    break

        for idx, line in enumerate(lines):
            if line.lower() == "--- page 3 ---":
                end_idx = idx
                break

        if start_idx is None:
            print("[BILLING][WARN] Could not find table header. No line items extracted.")
            return line_items

        if end_idx is None:
            end_idx = len(lines)

        table_lines = lines[start_idx:end_idx]
        print("[BILLING][DEBUG] table line count:", len(table_lines))

        i = 0
        while i + 3 < len(table_lines):
            label = table_lines[i]
            qty = table_lines[i + 1]
            rate = table_lines[i + 2]
            amount_line = table_lines[i + 3]

            label_lower = label.lower()

            # Validate table row structure:
            # label / qty-or-dash / rate-or-dash / amount
            qty_ok = qty == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", qty))
            rate_ok = rate == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", rate))
            amount_ok = bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", amount_line))

            if not (qty_ok and rate_ok and amount_ok):
                i += 1
                continue

            amount = self._parse_amount(amount_line)
            if amount < 10:
                i += 4
                continue

            category = self._match_category(label)
            raw_line = f"{label} {qty} {rate} {amount_line}"
            dedupe_key = (category, amount, re.sub(r"\s+", " ", label_lower).strip())

            if dedupe_key not in seen_entries:
                seen_entries.add(dedupe_key)
                print(f"[BILLING][ROW MATCHED] {category} | amount: {amount} | text: {raw_line}")
                line_items.append((category, amount, raw_line))

            i += 4

        return line_items

    def _build_breakdown(self, line_items: List[Tuple[str, float, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {}

        for category, amount, raw_line in line_items:
            final_category = category

            if final_category not in breakdown:
                breakdown[final_category] = {
                    "claimed": 0.0,
                    "approved": 0.0,
                    "status": "approved",
                    "reason": "Within expected range",
                    "lines": [],
                }

            breakdown[final_category]["claimed"] += amount
            breakdown[final_category]["lines"].append(raw_line)

            if final_category == "non_payable":
                breakdown[final_category]["status"] = "rejected"
                breakdown[final_category]["reason"] = "Non-medical or non-payable charge"
                print("[BILLING][FORCED NON-PAYABLE]", raw_line, "->", amount)
            else:
                breakdown[final_category]["approved"] += amount

        total_claim_amount = self._parse_amount(context.get("total_claim_amount"))

        for category, entry in breakdown.items():
            print("[BILLING][CATEGORY CHECK]", category, "claimed=", entry["claimed"])
            for line in entry["lines"]:
                print("   [LINE]", line)

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

        try:
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
            print(combined_text[:1500])
            print("[BILLING][DEBUG] combined_text repr:")
            print(repr(combined_text[:1500]))

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

            # Billing totals should reflect parsed bill rows
            effective_claimed_total = parsed_total if parsed_total > 0 else declared_total

            approved_total = approved_total_before_clamp
            if effective_claimed_total > 0:
                approved_total = min(approved_total, effective_claimed_total)

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
                "Extracted billing line items using row-based table parsing",
                "Grouped charges into billing categories",
                "Applied rule-based approval thresholds by category",
                f"Parsed total claimed: {parsed_total:.2f}",
                f"Approved total: {approved_total:.2f}",
                f"Anomaly score computed as {anomaly_score:.2f}",
            ]

            if "non_payable" in breakdown:
                reasoning.append(
                    f"Rejected non-payable charges totaling INR {breakdown['non_payable']['claimed']:.2f}"
                )

            if "room_charges" in breakdown and breakdown["room_charges"]["status"] == "partial":
                room_deduction = breakdown["room_charges"]["claimed"] - breakdown["room_charges"]["approved"]
                reasoning.append(
                    f"Applied room charge cap; reduced room charges by INR {room_deduction:.2f}"
                )
            if "doctor_fees" in breakdown and breakdown["doctor_fees"]["status"] == "partial":
                doctor_deduction = breakdown["doctor_fees"]["claimed"] - breakdown["doctor_fees"]["approved"]
                reasoning.append(
                    f"Applied doctor fee threshold; reduced doctor fees by INR {doctor_deduction:.2f}"
                )

            if "consumables" in breakdown and breakdown["consumables"]["status"] == "partial":
                consumable_deduction = breakdown["consumables"]["claimed"] - breakdown["consumables"]["approved"]
                reasoning.append(
                    f"Applied consumables threshold; reduced consumables by INR {consumable_deduction:.2f}"
                )
            if deductions > 0:
                reasoning.append(f"Total billing deductions computed as INR {deductions:.2f}")

            if anomaly_score > 0:
                reasoning.append(f"Billing anomaly score assessed at {anomaly_score:.2f}")
            reasoning.append(f"Parsed {len(line_items)} bill line items across {len(breakdown)} billing categories")
            reasoning.append(f"Billing categories identified: {', '.join(sorted(breakdown.keys()))}")
            output = {
                "total_claimed": effective_claimed_total,
                "total_approved": approved_total,
                "breakdown": breakdown,
                "anomaly_score": anomaly_score,
                "pricing_benchmark": "Rule-based benchmark v5",
                "deductions": deductions,
            }

            print("[BILLING][FINAL OUTPUT]", output)
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

        except Exception as e:
            elapsed = time.time() - start
            print("[BILLING][ERROR]", repr(e))
            logger.exception(f"[{self.AGENT_NAME}] Failed")

            return {
                "agent": self.AGENT_NAME,
                "owner": self.OWNER,
                "status": "failed",
                "reasoning": [f"Billing agent failed: {repr(e)}"],
                "output": {
                    "total_claimed": 0.0,
                    "total_approved": 0.0,
                    "breakdown": {},
                    "anomaly_score": 0.0,
                    "pricing_benchmark": "Rule-based benchmark v5",
                    "deductions": 0.0,
                },
                "confidence": 0.0,
                "processing_time": f"{elapsed:.1f}s",
            }
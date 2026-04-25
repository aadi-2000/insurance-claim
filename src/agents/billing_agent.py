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
        """Parse amount handling Indian number format (1,50,895.67)"""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        # Convert to string and remove currency symbols, spaces
        value_str = str(value).strip()
        
        # Remove currency symbols (₹, Rs, INR, etc.)
        value_str = re.sub(r'(?i)(rs\.?|inr|₹)\s*', '', value_str)
        
        # Remove commas (Indian format uses commas: 1,50,895.67)
        value_str = value_str.replace(',', '')
        
        # Remove any remaining non-digit characters except decimal point
        value_str = re.sub(r'[^\d.]', '', value_str)
        
        # Handle multiple decimal points (keep only the last one)
        if value_str.count('.') > 1:
            parts = value_str.split('.')
            value_str = ''.join(parts[:-1]) + '.' + parts[-1]
        
        try:
            return float(value_str) if value_str else 0.0
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
                "bedside monitoring",
                "oxygen support charges",
            ],
            "procedure_charges": [
                "procedure",
                "surgery",
                "operation",
                "operation theatre",
                "ot charges",
                "surgical",
                "appendectomy",
                "appendicectomy",
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
                "iv antibiotics",
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
                "chest x-ray",
                "hrct",
                "thorax",
            ],
            "consumables": [
                "consumable",
                "gloves",
                "syringe",
                "catheter",
                "dressing",
                "sutures",
                "respiratory consumables",
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
                "physician visit",
                "pulmonology specialist consultation",
            ],
            "non_payable": [
                "registration",
                "file",
                "file handling",
                "file processing",
                "welcome kit",
                "kit charge",
                "convenience kit",
                "patient kit",
                "patient amenity kit",
                "attender",
                "visitor meal",
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
            "visitor meal",
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

    def _has_billing_keyword(self, text: str) -> bool:
        lower = text.lower()
        category_keywords = self._get_category_keywords()

        flat_keywords = []
        for keywords in category_keywords.values():
            flat_keywords.extend(keywords)

        generic_keywords = [
            "charges",
            "charge",
            "amount",
            "bill",
            "receipt",
            "invoice",
            "fee",
            "fees",
        ]

        return any(k in lower for k in flat_keywords + generic_keywords)

    def _looks_like_date_or_metadata_amount(self, line: str, amount: float) -> bool:
        lower = line.lower()

        # Skip phone numbers (10 digits, typically > 1 million)
        if amount >= 1000000000:  # 10-digit numbers (phone numbers)
            if any(keyword in lower for keyword in ["contact", "phone", "tel", "call", "helpline", "mobile", "number"]):
                return True
            # If it's a very large number without currency symbols or billing keywords, likely a phone/reference number
            if not any(symbol in line for symbol in ["₹", "rs.", "rs ", "inr"]):
                return True
        
        # Skip reference/ID numbers (7-8 digits without decimal points)
        if 1000000 <= amount <= 99999999 and amount == int(amount):
            # Check if line contains ID/reference keywords
            if any(keyword in lower for keyword in ["id", "no.", "no ", "number", "ref", "code", "mrn", "uhid", "ip.no", "ipno"]):
                return True
            # If it's a whole number > 1M without currency context, likely an ID
            if not any(symbol in line for symbol in ["₹", "rs.", "rs ", "inr", "amount", "total", "charges"]):
                return True

        # Skip common years when the line looks like a date/treatment sentence, not a bill row
        if 1900 <= amount <= 2100:
            month_names = [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
                "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            ]
            if any(m in lower for m in month_names):
                return True
            if "date" in lower:
                return True

        metadata_keywords = [
            "patient name",
            "policy number",
            "uhid",
            "hospital name",
            "diagnosis",
            "primary diagnosis",
            "admission date",
            "discharge date",
            "claim status",
            "documents attached",
            "treatment details",
            "summary:",
            "clinical summary",
            "authorized signatory",
            "billing executive",
            "attending consultant",
            "ward category",
            "tpa / insurer",
            "mode of payment",
            "emergency contact",
            "lactation helpline",
        ]

        if any(k in lower for k in metadata_keywords):
            return True

        return False

    def _extract_table_line_items(self, lines: List[str]) -> List[Tuple[str, float, str]]:
        line_items: List[Tuple[str, float, str]] = []
        seen_entries = set()

        start_idx = None
        end_idx = None
        table_mode = None  # "five_col" or "four_col"

        # 5-column header:
        # Sl. / No. / Description / Qty/Days / Rate (INR) / Amount (INR)
        for idx in range(len(lines) - 5):
            window = [lines[idx + j].lower().strip() for j in range(6)]
            if (
                window[0] == "sl."
                and window[1] == "no."
                and window[2] == "description"
                and "qty" in window[3]
                and "rate" in window[4]
                and "amount" in window[5]
            ):
                start_idx = idx + 6
                table_mode = "five_col"
                break

        # 4-column header:
        # Description / Qty / Rate / Amount
        if start_idx is None:
            for idx in range(len(lines) - 3):
                window = [lines[idx + j].lower().strip() for j in range(4)]
                if (
                    window[0] == "description"
                    and "qty" in window[1]
                    and "rate" in window[2]
                    and "amount" in window[3]
                ):
                    start_idx = idx + 4
                    table_mode = "four_col"
                    break

        stop_markers = {
            "gross amount",
            "net payable",
            "claim amount submitted",
            "billing summary",
            "authorized signatory",
        }

        if start_idx is not None:
            for idx in range(start_idx, len(lines)):
                line_lower = lines[idx].lower().strip()
                if line_lower in stop_markers or line_lower.startswith("gross amount") or line_lower.startswith("net payable") or line_lower.startswith("claim amount submitted"):
                    end_idx = idx
                    break

        if start_idx is None:
            return line_items

        if end_idx is None:
            end_idx = len(lines)

        table_lines = lines[start_idx:end_idx]

        i = 0

        if table_mode == "five_col":
            while i + 4 < len(table_lines):
                serial_no = table_lines[i].strip()
                label = table_lines[i + 1].strip()
                qty = table_lines[i + 2].strip()
                rate = table_lines[i + 3].strip()
                amount_line = table_lines[i + 4].strip()

                serial_ok = bool(re.fullmatch(r"\d+", serial_no))
                qty_ok = qty == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", qty))
                rate_ok = rate == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", rate))
                amount_ok = bool(re.fullmatch(r"-?[\d,]+(?:\.\d{1,2})?", amount_line))

                if not (serial_ok and qty_ok and rate_ok and amount_ok):
                    i += 1
                    continue

                amount = self._parse_amount(amount_line)
                if amount < 10:
                    i += 5
                    continue

                category = self._match_category(label)
                raw_line = f"{serial_no} {label} {qty} {rate} {amount_line}"
                dedupe_key = (category, amount, re.sub(r"\s+", " ", label.lower()).strip())

                if dedupe_key not in seen_entries:
                    seen_entries.add(dedupe_key)
                    line_items.append((category, amount, raw_line))

                i += 5

        elif table_mode == "four_col":
            while i + 3 < len(table_lines):
                label = table_lines[i].strip()
                qty = table_lines[i + 1].strip()
                rate = table_lines[i + 2].strip()
                amount_line = table_lines[i + 3].strip()

                qty_ok = qty == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", qty))
                rate_ok = rate == "-" or bool(re.fullmatch(r"[\d,]+(?:\.\d{1,2})?", rate))
                amount_ok = bool(re.fullmatch(r"-?[\d,]+(?:\.\d{1,2})?", amount_line))

                if not (qty_ok and rate_ok and amount_ok):
                    i += 1
                    continue

                amount = self._parse_amount(amount_line)
                if amount < 10:
                    i += 4
                    continue

                category = self._match_category(label)
                raw_line = f"{label} {qty} {rate} {amount_line}"
                dedupe_key = (category, amount, re.sub(r"\s+", " ", label.lower()).strip())

                if dedupe_key not in seen_entries:
                    seen_entries.add(dedupe_key)
                    line_items.append((category, amount, raw_line))

                i += 4

        return line_items

    def _extract_simple_line_items(self, lines: List[str]) -> List[Tuple[str, float, str]]:
        line_items: List[Tuple[str, float, str]] = []
        seen_entries = set()

        skip_keywords = [
            "--- page",
            "billing summary",
            "authorized signatory",
            "patient name",
            "policy number",
            "uhid",
            "diagnosis",
            "primary diagnosis",
            "admission date",
            "discharge date",
            "summary:",
            "clinical summary",
            "gross amount",
            "discount",
            "net payable amount",
            "claim amount submitted",
            "insurance claim document",
            "documents attached",
            "claim status",
            "treatment details",
            "hospital name",
            "ward category",
            "tpa / insurer",
            "mode of payment",
            "billing executive",
            "attending consultant",
        ]

        for line in lines:
            lower = line.lower().strip()

            if not lower or any(k in lower for k in skip_keywords):
                continue

            amount_matches = re.findall(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)", lower)
            if not amount_matches:
                continue

            amount = self._parse_amount(amount_matches[-1])
            if amount < 10:
                continue
            
            # Skip unreasonably large amounts (likely phone numbers or IDs)
            # Max reasonable single line item: 10 lakhs (1,000,000)
            if amount > 1000000:
                continue

            if self._looks_like_date_or_metadata_amount(line, amount):
                continue

            # Very important: only treat simple one-line rows as billing if they look like billing text
            if not self._has_billing_keyword(line):
                continue

            category = self._match_category(line)
            if category == "miscellaneous":
                continue

            dedupe_key = (category, amount, re.sub(r"\s+", " ", lower).strip())
            if dedupe_key in seen_entries:
                continue

            seen_entries.add(dedupe_key)
            line_items.append((category, amount, line))

        return line_items

    def _extract_line_items(self, text: str) -> List[Tuple[str, float, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        line_items = self._extract_table_line_items(lines)
        if line_items:
            return line_items
        return self._extract_simple_line_items(lines)

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
            else:
                breakdown[final_category]["approved"] += amount

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

            entry["approved"] = min(entry["approved"], entry["claimed"])

        return breakdown

    def _score_anomalies(self, breakdown: Dict[str, Any], declared_total: float) -> float:
        score = 0.0

        total_claimed = sum(item["claimed"] for item in breakdown.values())
        total_approved = sum(item["approved"] for item in breakdown.values())

        if declared_total > 0 and total_claimed > 0:
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
    
    def _extract_preferred_total(self, text: str) -> float:
        text_lower = text.lower()

        # STRICT priority order
        priority_keywords = [
            "claim amount submitted",
            "total claim amount",
            "net payable",
            "gross amount",
        ]

        for keyword in priority_keywords:
            for line in text_lower.splitlines():
                if keyword in line:
                    matches = re.findall(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)", line)
                    if matches:
                        value = self._parse_amount(matches[-1])
                        return value

        return 0.0
    
    async def _extract_billing_with_llm(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to extract billing breakdown from document text"""
        if not self.llm_client:
            return None
        
        # Smart truncation: prioritize billing-related sections
        max_chars = 20000  # Increased limit to capture full billing breakdown
        if len(text) > max_chars:
            # Look for billing keywords to find the billing section
            billing_keywords = ["FINAL BILL", "BILL BREAK", "GROSS AMOUNT", "NET PAYABLE", "PHARMACY", "LAB", "ROOM RENT"]
            lines = text.split('\n')
            
            # Find billing section
            billing_start = 0
            for i, line in enumerate(lines):
                if any(keyword in line.upper() for keyword in billing_keywords):
                    billing_start = max(0, i - 10)  # Include 10 lines before for context
                    break
            
            # Take from billing section onwards, up to max_chars
            billing_text = '\n'.join(lines[billing_start:])
            if len(billing_text) > max_chars:
                text = billing_text[:max_chars]
            else:
                text = billing_text
        
        
        prompt = f"""You are extracting billing information from a medical claim bill. This document contains itemized charges across multiple categories.

Document:
{text}

Extract ALL billing line items and categorize them into the following JSON format:
{{
  "total_claimed": <total gross/net payable amount as number>,
  "breakdown": {{
    "room_charges": <sum of all room/bed/ICU charges>,
    "doctor_fees": <sum of all doctor/consultant fees>,
    "investigations": <sum of all lab tests, scans, ECG, X-ray charges>,
    "medication": <sum of all pharmacy/medicine charges>,
    "consumables": <sum of all consumables, syringes, dressings>,
    "procedure_charges": <sum of all surgical/procedure/OT charges>,
    "other": <sum of any other charges not fitting above categories>
  }}
}}

IMPORTANT INSTRUCTIONS:
1. Look through the ENTIRE document for billing line items
2. Common keywords to look for:
   - Room: "Room Rent", "Bed Charges", "ICU", "Ward"
   - Doctor: "Doctor Fee", "Consultant", "Professional Fee"
   - Investigations: "Lab", "Laboratory", "Pathology", "Radiology", "ECG", "Scan", "X-Ray"
   - Medication: "Pharmacy", "Medicine", "Drug", "Injection"
   - Consumables: "Consumables", "Syringe", "Dressing", "Gloves", "Catheter"
   - Procedure: "Surgery", "OT Charges", "Operation Theatre", "Procedure"
3. Sum up ALL amounts in each category
4. The total_claimed should match "Gross Amount" or "Net Payable" or "Total Claim Amount"
5. Return amounts as numbers only (no ₹, Rs, or commas)
6. If a category has no charges, use 0

Return ONLY valid JSON, no explanation or markdown."""

        try:
            response = await self.llm_client.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.get("parsed")
            if result:
                logger.info(f"[BILLING] LLM extraction successful: {result}")
                return result
            else:
                logger.warning("[BILLING] LLM extraction returned no parsed result")
                return None
        except Exception as e:
            logger.error(f"[BILLING] LLM extraction failed: {e}")
            return None
    

    async def process(self, claim_data: Dict) -> Dict[str, Any]:
        start = time.time()

        try:

            inputs = self._collect_inputs(claim_data)
            extracted_requirements = inputs["extracted_requirements"]
            combined_text = inputs["combined_text"]


            # Try LLM extraction first
            context = {
                "total_claim_amount": extracted_requirements.get("total_claim_amount"),
                "diagnosis": extracted_requirements.get("diagnosis"),
                "hospital_name": extracted_requirements.get("hospital_name"),
                "admission_date": extracted_requirements.get("admission_date"),
                "discharge_date": extracted_requirements.get("discharge_date"),
            }
            
            llm_result = await self._extract_billing_with_llm(combined_text, context) if self.llm_client else None
            
            if llm_result and llm_result.get("breakdown"):
                breakdown = {}
                for category, amount in llm_result.get("breakdown", {}).items():
                    if amount > 0:
                        breakdown[category] = {
                            "claimed": float(amount),
                            "approved": float(amount),
                            "status": "approved",
                            "reason": "Within expected range",
                            "lines": [f"LLM extracted {category}: {amount}"]
                        }
                
                # Use declared total from requirements agent (more reliable than LLM extraction from OCR text)
                declared_total = self._parse_amount(extracted_requirements.get("total_claim_amount"))
                                
                parsed_total = sum(item["claimed"] for item in breakdown.values())
                
                # Use declared_total as the official claimed amount (from bill's gross total)
                effective_claimed_total = declared_total
                
                # Apply category limits without scaling
                total_claim_amount = declared_total
                for category, entry in breakdown.items():
                    claimed = entry["claimed"]
                    
                    if category == "room_charges" and claimed > 50000:
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
                
                # Calculate approved total from breakdown (no scaling)
                approved_total = sum(item["approved"] for item in breakdown.values())
                deductions = effective_claimed_total - approved_total
                anomaly_score = self._score_anomalies(breakdown, declared_total)
                
                billing_summary = {
                    "line_items_parsed": len(breakdown),
                    "categories_detected": list(breakdown.keys()),
                    "non_payable_total": 0.0,
                    "category_deductions": {},
                    "deduction_reason_summary": ["LLM-based extraction"],
                }
                
                output = {
                    "total_claimed": effective_claimed_total,
                    "total_approved": approved_total,
                    "breakdown": breakdown,
                    "anomaly_score": anomaly_score,
                    "pricing_benchmark": "LLM extraction with rule-based validation",
                    "deductions": deductions,
                    "billing_summary": billing_summary,
                }
                
                elapsed = time.time() - start
                return {
                    "agent": self.AGENT_NAME,
                    "owner": self.OWNER,
                    "status": "success",
                    "reasoning": [f"LLM extracted billing breakdown with {len(breakdown)} categories"],
                    "output": output,
                    "confidence": 0.92,
                    "processing_time": f"{elapsed:.1f}s",
                }
            
            # Fallback to regex parsing if LLM fails
            line_items = self._extract_line_items(combined_text)

            declared_total_preview = self._parse_amount(extracted_requirements.get("total_claim_amount"))

            # Fallback for claim-summary documents with declared total but no itemized bill rows
            if len(line_items) == 0 and declared_total_preview > 0:

                billing_summary = {
                    "line_items_parsed": 0,
                    "categories_detected": [],
                    "non_payable_total": 0.0,
                    "category_deductions": {},
                    "deduction_reason_summary": [
                        "No itemized billing found, used declared claim amount",
                        "No deductions applied due to lack of bill-level detail"
                    ],
                }

                output = {
                    "total_claimed": declared_total_preview,
                    "total_approved": declared_total_preview,
                    "breakdown": {},
                    "anomaly_score": 0.05,
                    "pricing_benchmark": "Declared-total fallback",
                    "deductions": 0.0,
                    "billing_summary": billing_summary,
                }
                reasoning = [
                    "No itemized billing rows were found in the uploaded claim document.",
                    f"Used declared claim amount fallback of INR {declared_total_preview:.2f}.",
                    "Approved amount was kept equal to claimed amount because no bill-line evidence was available for deductions.",
                    "Billing anomaly score kept low due to absence of contradictory itemized billing data.",
                ]

                elapsed = time.time() - start

                return {
                    "agent": self.AGENT_NAME,
                    "owner": self.OWNER,
                    "status": "success",
                    "reasoning": reasoning,
                    "output": output,
                    "confidence": 0.85,
                    "processing_time": f"{elapsed:.1f}s",
                }

            context = {
                "total_claim_amount": extracted_requirements.get("total_claim_amount"),
                "diagnosis": extracted_requirements.get("diagnosis"),
                "hospital_name": extracted_requirements.get("hospital_name"),
                "admission_date": extracted_requirements.get("admission_date"),
                "discharge_date": extracted_requirements.get("discharge_date"),
            }

            breakdown = self._build_breakdown(line_items, context)

            declared_total = self._extract_preferred_total(combined_text)
            if declared_total == 0:
                declared_total = self._parse_amount(extracted_requirements.get("total_claim_amount"))

            parsed_total = sum(item["claimed"] for item in breakdown.values())
            approved_total_before_clamp = sum(item["approved"] for item in breakdown.values())

            effective_claimed_total = parsed_total if parsed_total > 0 else declared_total

            approved_total = approved_total_before_clamp
            if effective_claimed_total > 0:
                approved_total = min(approved_total, effective_claimed_total)

            deductions = max(effective_claimed_total - approved_total, 0.0)
            anomaly_score = self._score_anomalies(breakdown, declared_total)

            reasoning = []

            reasoning.append(f"Parsed {len(line_items)} billing line items from uploaded documents.")

            if breakdown:
                reasoning.append(
                    f"Identified billing categories: {', '.join(k.replace('_', ' ') for k in sorted(breakdown.keys()))}."
                )
            else:
                reasoning.append("No billing categories were identified from itemized lines.")

            reasoning.append(
                f"Computed claimed amount as INR {effective_claimed_total:.2f} and approved amount as INR {approved_total:.2f}."
            )

            for category in sorted(breakdown.keys()):
                entry = breakdown[category]
                claimed = entry.get("claimed", 0.0)
                approved = entry.get("approved", 0.0)
                status = entry.get("status", "approved")
                reason = entry.get("reason", "")

                if status == "rejected":
                    reasoning.append(
                        f"{category.replace('_', ' ').title()} rejected: claimed INR {claimed:.2f}, approved INR {approved:.2f}. Reason: {reason}."
                    )
                elif status == "partial":
                    reduction = claimed - approved
                    reasoning.append(
                        f"{category.replace('_', ' ').title()} partially approved: claimed INR {claimed:.2f}, approved INR {approved:.2f}, reduction INR {reduction:.2f}. Reason: {reason}."
                    )
                else:
                    reasoning.append(
                        f"{category.replace('_', ' ').title()} approved in full at INR {approved:.2f}."
                    )

            if deductions > 0:
                reasoning.append(f"Total deductions applied: INR {deductions:.2f}.")
            else:
                reasoning.append("No deductions were applied.")

            if anomaly_score >= 0.5:
                reasoning.append(
                    f"Billing anomaly score is high at {anomaly_score:.2f}, indicating significant mismatch or adjustment."
                )
            elif anomaly_score >= 0.2:
                reasoning.append(
                    f"Billing anomaly score is moderate at {anomaly_score:.2f}, reflecting some adjustments in billing review."
                )
            else:
                reasoning.append(
                    f"Billing anomaly score is low at {anomaly_score:.2f}, indicating the bill is broadly consistent."
                )


            non_payable_total = breakdown.get("non_payable", {}).get("claimed", 0.0)

            partial_deductions = {}
            for category, entry in breakdown.items():
                claimed = entry.get("claimed", 0.0)
                approved = entry.get("approved", 0.0)
                if approved < claimed:
                    partial_deductions[category] = round(claimed - approved, 2)

            billing_summary = {
                "line_items_parsed": len(line_items),
                "categories_detected": sorted(list(breakdown.keys())),
                "non_payable_total": round(non_payable_total, 2),
                "category_deductions": partial_deductions,
                "deduction_reason_summary": [],
            }

            if non_payable_total > 0:
                billing_summary["deduction_reason_summary"].append(
                    f"Rejected non-payable charges worth INR {non_payable_total:.2f}"
                )

            if "room_charges" in partial_deductions:
                billing_summary["deduction_reason_summary"].append(
                    f"Room charges reduced by INR {partial_deductions['room_charges']:.2f} due to cap"
                )

            if "doctor_fees" in partial_deductions:
                billing_summary["deduction_reason_summary"].append(
                    f"Doctor fees reduced by INR {partial_deductions['doctor_fees']:.2f} due to threshold"
                )

            if "consumables" in partial_deductions:
                billing_summary["deduction_reason_summary"].append(
                    f"Consumables reduced by INR {partial_deductions['consumables']:.2f} due to threshold"
                )

            if not billing_summary["deduction_reason_summary"]:
                billing_summary["deduction_reason_summary"].append("No billing deductions applied")

            output = {
                "total_claimed": effective_claimed_total,
                "total_approved": approved_total,
                "breakdown": breakdown,
                "anomaly_score": anomaly_score,
                "pricing_benchmark": "Rule-based benchmark v6",
                "deductions": deductions,
                "billing_summary": billing_summary,
            }

            await asyncio.sleep(0.1)

            elapsed = time.time() - start

            # --- FINAL STRONG CONFIDENCE LOGIC ---
            base_conf = 1.0 - (anomaly_score * 0.3)
            line_score = min(len(line_items) / 8, 1.0)
            category_score = min(len(breakdown) / 5, 1.0)

            consistency_score = 1.0
            if declared_total > 0 and parsed_total > 0:
                mismatch_ratio = abs(parsed_total - declared_total) / declared_total
                if mismatch_ratio > 0.2:
                    consistency_score = 0.75
                elif mismatch_ratio > 0.1:
                    consistency_score = 0.9

            confidence = (
                base_conf * 0.4 +
                line_score * 0.3 +
                category_score * 0.2 +
                consistency_score * 0.1
            )

            confidence = max(0.85, min(confidence, 0.98))


            return {
                "agent": self.AGENT_NAME,
                "owner": self.OWNER,
                "status": "success",
                "reasoning": reasoning,
                "output": output,
                "confidence": confidence,
                "processing_time": f"{elapsed:.1f}s",
            }

        except Exception as e:
            elapsed = time.time() - start
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
                    "pricing_benchmark": "Rule-based benchmark v6",
                    "deductions": 0.0,
                },
                "confidence": 0.0,
                "processing_time": f"{elapsed:.1f}s",
            }
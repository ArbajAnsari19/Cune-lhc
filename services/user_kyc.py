import os
import json
import re
from typing import List, Dict, Any
from config.settings import OUTPUT_DIR


# key variants used across the function (lower-case)
_KEY_VARIANTS = {
    "aadhaar_name": ["name_english", "name", "name_hindi", "full_name"],
    "aadhaar_number": ["aadhaar_number", "aadhaar no", "uid", "uidai_number"],
    "aadhaar_address": [
        "address_english",
        "address",
        "address_line1_english",
        "address_line1",
        "address_line2",
        "address_line3",
    ],
    "aadhaar_father": ["s_o_name_english", "s_o_name", "s_o", "father_or_husband_name", "father_name"],

    "voter_name": ["name_english", "name"],
    "voter_father": ["father_s_name", "father_name"],
    "voter_address": ["address", "address_english"],
    "voter_dob": ["date_of_birth", "dob"],

    "pan_name": ["name", "name_english"],
    # ✅ Added `permanent_account_number` variant so PAN Number extracts correctly
    "pan_number": ["pan_number", "pan no", "pan", "permanent_account_number"],
    "pan_father": ["father_name"],
    "pan_dob": ["date_of_birth", "dob"],

    "dl_name": ["name", "name_english"],
    "dl_number": [
        "dl_number",
        "dl_no",
        "driving_licence_number",
        "driving_license_number",
        "licence_number",
    ],
    "dl_father": ["father_or_husband_name", "father_name"],
    "dl_address": ["address"],

    "passport_given": ["given_names", "given_name"],
    "passport_surname": ["surname", "family_name"],
    "passport_name": ["name"],
    "passport_dob": ["date_of_birth", "dob"],
    "passport_address_parts": [
        "address_building",
        "address_city",
        "address_state",
        "address_country",
    ],
}



def _load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return list of dict records (each with lower-cased keys)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    records: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # normalize keys to lower-case for robust lookups
                records.append({str(k).lower(): v for k, v in item.items()})
    return records


def _first_from_records(records: List[Dict[str, Any]], variants: List[str]):
    """Return first non-empty value found for any variant key across records (variants expected lower-case)."""
    for rec in records:
        for v in variants:
            val = rec.get(v)
            if val not in (None, "", []):
                return val
    return None


def _extract_father_from_aadhaar_address(records: List[Dict[str, Any]]):
    """
    Try to extract father name from Aadhaar address fields using common 'S/O' patterns.
    Looks in address_english or address fields across records.
    """
    so_pattern = re.compile(r"\bS(?:/|\\s*\/?\\s*)O[:\s]*([^,\\n]+)", flags=re.IGNORECASE)
    for rec in records:
        for addr_key in ("address_english", "address"):
            addr = rec.get(addr_key)
            if isinstance(addr, str) and addr.strip():
                # try S/O or S O or S/O:
                m = so_pattern.search(addr)
                if m:
                    name = m.group(1).strip()
                    # trim trailing tokens like 'S/O: FOO BAR' -> take up to comma or - or '('
                    name = re.split(r"[,|\-|\\(|/]", name)[0].strip()
                    if name:
                        return name
    # fallback: check standard father keys
    return _first_from_records(records, _KEY_VARIANTS["aadhaar_father"])


def generate_user_kyc(base_dir: str | None = None) -> None:
    """
    Consolidate KYC fields from documents in verification_documents into user_kyc.json.

    Priority:
      Full Name: Aadhaar > Voter > PAN > DL > Passport
      Father Name: Aadhaar > Voter > PAN > DL > Passport
      Aadhaar Number: Aadhaar only
      PAN Number: PAN only
      DL No.: Driving Licence only
      DOB: Passport > Aadhaar > Voter > PAN > DL
      Address: Aadhaar > Voter > PAN > DL > Passport
    """
    if base_dir is None:
        base_root = os.path.abspath(os.path.join(OUTPUT_DIR, os.pardir))
        base_dir = os.path.join(base_root, "verification_documents")

    os.makedirs(base_dir, exist_ok=True)

    # Load records (normalized lower-case keys)
    aadhaar_records = _load_records(os.path.join(base_dir, "aadhaar.json"))
    voter_records = _load_records(os.path.join(base_dir, "voter.json"))
    pan_records = _load_records(os.path.join(base_dir, "pan.json"))
    dl_records = _load_records(os.path.join(base_dir, "driving_licence.json"))
    passport_records = _load_records(os.path.join(base_dir, "passport.json"))

    # Helper to assemble passport full name
    def _passport_fullname():
        # prefer given_names + surname
        given = _first_from_records(passport_records, _KEY_VARIANTS["passport_given"])
        surname = _first_from_records(passport_records, _KEY_VARIANTS["passport_surname"])
        if given:
            if surname:
                return f"{given} {surname}".strip()
            return given
        # fallback to single name field
        return _first_from_records(passport_records, _KEY_VARIANTS["passport_name"])

    # Full Name priority: Aadhaar > Voter > PAN > DL > Passport
    full_name = (
        _first_from_records(aadhaar_records, _KEY_VARIANTS["aadhaar_name"])
        or _first_from_records(voter_records, _KEY_VARIANTS["voter_name"])
        or _first_from_records(pan_records, _KEY_VARIANTS["pan_name"])
        or _first_from_records(dl_records, _KEY_VARIANTS["dl_name"])
        or _passport_fullname()
    )

    # Father Name priority: Aadhaar (S/O extraction) > Voter > PAN > DL > Passport
    father_name = (
        _extract_father_from_aadhaar_address(aadhaar_records)
        or _first_from_records(voter_records, _KEY_VARIANTS["voter_father"])
        or _first_from_records(pan_records, _KEY_VARIANTS["pan_father"])
        or _first_from_records(dl_records, _KEY_VARIANTS["dl_father"])
        or _first_from_records(passport_records, ["father_legal_guardian_name", "father_name"])
    )

    # Aadhaar number (only from aadhaar file)
    aadhaar_number = _first_from_records(aadhaar_records, _KEY_VARIANTS["aadhaar_number"])

    # PAN number
    pan_number = _first_from_records(pan_records, _KEY_VARIANTS["pan_number"])

    # DL number: check driving_licence file variants
    dl_number = _first_from_records(dl_records, _KEY_VARIANTS["dl_number"])

    # Date of birth priority: Passport > Aadhaar > Voter > PAN > DL
    dob = (
        _first_from_records(passport_records, _KEY_VARIANTS["passport_dob"])
        or _first_from_records(aadhaar_records, ["date_of_birth", "dob", "birth_date"])
        or _first_from_records(voter_records, _KEY_VARIANTS["voter_dob"])
        or _first_from_records(pan_records, _KEY_VARIANTS["pan_dob"])
        or _first_from_records(dl_records, ["date_of_birth", "dob"])
    )

    # Address priority: Aadhaar > Voter > PAN > DL > Passport (passport assembled from parts)
    passport_addr = None
    if passport_records:
        parts = []
        for p in _KEY_VARIANTS["passport_address_parts"]:
            part = _first_from_records(passport_records, [p])
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
        passport_addr = ", ".join(parts) if parts else None

    address = (
        _first_from_records(aadhaar_records, _KEY_VARIANTS["aadhaar_address"])
        or _first_from_records(voter_records, _KEY_VARIANTS["voter_address"])
        or _first_from_records(pan_records, ["address"])
        or _first_from_records(dl_records, _KEY_VARIANTS["dl_address"])
        or passport_addr
    )

    # Build KYC object
    kyc = {
        "Full Name": full_name,
        "Father Name": father_name,
        "Aadhaar Number": aadhaar_number,
        "PAN Number": pan_number,
        "DL Number": dl_number,
        "Date of Birth": dob,
        "Address": address,
    }

    # Persist user_kyc.json (silent)
    out_path = os.path.join(base_dir, "user_kyc.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(kyc, fh, indent=2, ensure_ascii=False)
    except Exception:
        # keep function silent on error as requested
        pass

import os
import json
import shutil
from config.settings import OUTPUT_DIR


def classify_verification_documents(base_dir: str = OUTPUT_DIR, target_dir: str | None = None) -> None:
    """
    Robustly classify JSON files in `base_dir` and move them into `verification_documents`
    (sibling of outputs) using fixed names:
       aadhaar.json, passport.json, voter.json, driving_licence.json, pan.json

    Behavior:
    - `target_dir` defaults to sibling of OUTPUT_DIR if not provided
    - Silent operation (no prints) and no return value
    - Gracefully skips invalid/unsupported files
    """

    if target_dir is None:
        base_root = os.path.abspath(os.path.join(base_dir, os.pardir))
        target_dir = os.path.join(base_root, "verification_documents")

    os.makedirs(target_dir, exist_ok=True)

    DOC_FILENAME = {
        "aadhaar": "aadhaar.json",
        "passport": "passport.json",
        "voter": "voter.json",
        "driving_licence": "driving_licence.json",
        "pan": "pan.json",
    }

    # Helper: flatten list/dict -> list of dict records
    def as_record_list(data):
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    # Lower-case keys map
    def lower_keys(d):
        return {str(k).lower(): v for k, v in d.items()}

    KEY_VARIANTS = {
        "aadhaar": [
            "aadhaar_number", "aadhaar no", "aadhaar card", "uid", "uidai", "uidai_number",
            "aadhaarid", "aadhaarid_number", "aadhaaridno", "unique identification number", 
            "aadhaar_ref", "aadhaar reference"
        ],
        "passport": [
            "passport_number", "passport no", "passport", "passport card", "document_number", 
            "passportid", "passportid_number", "passportidno", "passport reference", 
            "passport code"
        ],
        "voter": [
            "epic_number", "epic no", "voter_id", "voterid", "voterid_number", "voteridno",
            "voter id no", "voter card", "voter_card", "voteridcard", "voter reference",
            "eci number", "election commission", "epic", "epic card"
        ],
        "driving_licence": [
            "dl_number", "dl_no", "driving_licence_number", "driving_license_number",
            "licence_number", "licence_no", "drivinglicence_number", "drivinglicence_no",
            "drivinglicence", "dlid", "dlid_number", "dlidno", "driver's license", 
            "driver licence", "driving licence card", "driving license card"
        ],
        "pan": [
            "pan_number", "pan no", "pan", "pan card", "pan id", "panid", "panid_number",
            "panidno", "permanent_account_number", "pan ref", "pan reference"
        ],
    }


    # Scoring function that includes many key variants and document_type text matches
    def score_record(rec):
        s = {"aadhaar": 0, "passport": 0, "voter": 0, "driving_licence": 0, "pan": 0}

        # convenience checks
        def has_any(keys):
            for k in keys:
                if rec.get(k) not in (None, "", []):
                    return True
            return False

        # document_type textual hints
        doc_type_text = (rec.get("document_type") or "").lower()

        # Aadhaar scoring
        if has_any(KEY_VARIANTS["aadhaar"]):
            s["aadhaar"] += 12
        if has_any(["name", "name_english", "name_hindi", "full_name"]):
            s["aadhaar"] += 1
        if has_any(["s_o_name_english", "s_o_name_hindi", "s_o", "s/o", "father_or_husband_name", "father_name"]):
            s["aadhaar"] += 1
        if has_any(["address", "address_english", "address_line1_english", "address_line1"]):
            s["aadhaar"] += 1
        if has_any(["dob", "date_of_birth", "birth_date"]):
            s["aadhaar"] += 1

        # Passport scoring
        if has_any(KEY_VARIANTS["passport"]) or "mrz_line1" in rec or "mrz_line2" in rec:
            s["passport"] += 12
        if has_any(["given_names", "surname", "name"]):
            s["passport"] += 1
        if has_any(["place_of_issue", "place_of_birth", "address_city"]):
            s["passport"] += 1
        if has_any(["dob", "date_of_birth", "birth_date"]):
            s["passport"] += 1
        if "passport" in doc_type_text or "p<" in (rec.get("mrz_line1") or "").lower():
            s["passport"] += 2

        # Voter scoring
        if has_any(KEY_VARIANTS["voter"]):
            s["voter"] += 12
        if has_any(["assembly_constituency_name", "part_number", "part_name"]):
            s["voter"] += 1
        if has_any(["name", "name_english"]):
            s["voter"] += 1
        if has_any(["address", "address_english"]):
            s["voter"] += 1
        if has_any(["dob", "date_of_birth"]):
            s["voter"] += 1
        if "elector" in doc_type_text or "voter" in doc_type_text or "epic" in doc_type_text:
            s["voter"] += 1

        # Driving licence scoring - expanded key variants and document_type match
        if has_any(KEY_VARIANTS["driving_licence"]):
            s["driving_licence"] += 12
        if "driving" in doc_type_text or "licence" in doc_type_text or "driving licence" in doc_type_text:
            s["driving_licence"] += 4
        if has_any(["class_lmv_issued_on", "class_mcwg_issued_on", "issued_on_lmv", "issued_on_mcwg", "issued_on"]):
            s["driving_licence"] += 1
        if has_any(["issuing_authority", "issuing_authority_signature", "issuing_authority_name"]):
            s["driving_licence"] += 1
        if has_any(["name", "name_english", "name_hindi"]):
            s["driving_licence"] += 1
        if has_any(["dob", "date_of_birth", "birth_date"]):
            s["driving_licence"] += 1
        if has_any(["state", "issuing_state", "state_code"]):
            s["driving_licence"] += 1

        # PAN scoring
        if has_any(KEY_VARIANTS["pan"]):
            s["pan"] += 12
        if has_any(["name", "name_english"]):
            s["pan"] += 1
        if has_any(["father_name", "father_or_husband_name"]):
            s["pan"] += 1
        if has_any(["dob", "date_of_birth"]):
            s["pan"] += 1

        return s

    # Decide file classification using aggregated scores across all records
    def classify_file(records):
        agg = {"aadhaar": 0, "passport": 0, "voter": 0, "driving_licence": 0, "pan": 0}
        primaries_present = {k: False for k in agg}

        for r in records:
            if not isinstance(r, dict):
                continue
            rn = lower_keys(r)
            sc = score_record(rn)
            for k in agg:
                agg[k] += sc[k]

            # primaries presence checks (use key variants + doc_type text)
            for k, variants in KEY_VARIANTS.items():
                if any(rn.get(v) not in (None, "", []) for v in variants):
                    primaries_present[k] = True

            # document_type textual hints
            dt = (rn.get("document_type") or "").lower()
            if "driving" in dt or "licence" in dt:
                primaries_present["driving_licence"] = True
            if "passport" in dt:
                primaries_present["passport"] = True
            if "elector" in dt or "voter" in dt or "epic" in dt:
                primaries_present["voter"] = True
            if "aadhaar" in dt or "uidai" in dt:
                primaries_present["aadhaar"] = True
            if "pan" in dt or "income tax" in dt:
                primaries_present["pan"] = True

        # choose best type by aggregated score
        best_type, best_score = max(agg.items(), key=lambda it: it[1])

        # Conservative thresholds:
        # - If primary identifier present and agg score >= 12 -> accept
        # - Else if agg score >= 10 -> accept
        # - Otherwise reject (None)
        if primaries_present.get(best_type) and best_score >= 12:
            return best_type
        if best_score >= 10:
            return best_type
        return None

    # iterate files silently
    for file_name in os.listdir(base_dir):
        if not file_name.lower().endswith(".json"):
            continue

        in_path = os.path.join(base_dir, file_name)
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            # invalid JSON -> skip
            continue

        records = as_record_list(data)
        if not records:
            continue

        doc_type = classify_file(records)
        if not doc_type:
            continue

        target_fname = DOC_FILENAME.get(doc_type)
        if not target_fname:
            continue

        dest_path = os.path.join(target_dir, target_fname)
        try:
            # remove existing target if possible (overwrite behavior)
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    # cannot remove existing target: skip to avoid clobber
                    continue
            shutil.move(in_path, dest_path)
        except Exception:
            # skip any move errors silently
            continue

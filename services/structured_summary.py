import json
from typing import Literal

from services.analyze import analyze_with_claude, extract_json_from_response


def build_structured_prompt(insurance_type: Literal["life", "property_casualty"], extracted_data: dict) -> str:
    data_str = json.dumps(extracted_data, indent=2)

    if insurance_type == "life":
        schema = {
            "applicant": {
                "name": "string | null",
                "dateOfBirth": "string | null",
                "fathersName": "string | null",
                "address": "string | null"
            },
            "policyDetails": {
                "policyType": "string | null",
                "coverageAmount": "string | null",
                "planName": "string | null",
                "policyTerms": "string | null",
                "renewalBasis": "string | null"
            },
            "medicalDisclosure": {
                "medicalHistory": "string | null",
                "familyHistory": "string | null",
                "weight": "string | null",
                "height": "string | null",
                "currentMedication": "string | null"
            },
            "lifestyleAssessment": {
                "smokingStatus": "string | null",
                "alcoholConsumption": "string | null",
                "physicalActivity": "string | null",
                "otherHabits": "string | null"
            }
        }
        instructions = (
            "Extract the requested life insurance applicant, policy details, medical disclosure, "
            "and lifestyle assessment fields from the provided data. If a field is not found, use null."
        )
    else:
        schema = {
            "applicant": {
                "name": "string | null",
                "dateOfBirth": "string | null",
                "fathersName": "string | null",
                "address": "string | null"
            },
            "propertyDetails": {
                "propertyAddress": "string | null",
                "propertyType": "string | null",
                "constructionType": "string | null",
                "roofType": "string | null",
                "numberOfStories": "int | null",
                "yearBuilt": "int | null",
                "extras": "object | null (any additional details like etc.)"
            },
            "propertyFeatures": {
                "hasCentralAir": "bool | null",
                "hasFireExtinguisher": "bool | null",
                "hasSwimmingPool": "bool | null",
                "extras": "object | null (any additional features like etc.)"
            },
            "riskFactors": {
                "distanceToFireStation": "string | null",
                "floodZone": "string | null",
                "hasPreviousClaim": "bool | null",
                "extras": "object | null (any additional risk factors like etc.)"
            }
        }
        instructions = (
            "Extract the requested property & casualty applicant, property details, property features, "
            "and risk factors from the provided data. If a field is not found, use null."
        )

    return (
        f"You are an expert underwriter assistant. {instructions}\n\n"
        f"Return ONLY a JSON object strictly matching the structure below. Do NOT include any extra text.\n\n"
        f"Schema (types are illustrative, keep the exact keys):\n{json.dumps(schema, indent=2)}\n\n"
        f"<extracted_data>\n{data_str}\n</extracted_data>\n"
    )


async def run_structured_summary_prompt(insurance_type: Literal["life", "property_casualty"], extracted_data: dict) -> dict:
    prompt = build_structured_prompt(insurance_type, extracted_data)
    result = await analyze_with_claude(prompt)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Claude analysis failed"))
    parsed = extract_json_from_response(result["analysis"])
    return parsed


def build_consolidation_prompt(
    insurance_type: Literal["life", "property_casualty"],
    per_file_summaries: list[dict]
) -> str:
    """Build a strict prompt to merge multiple structured summaries into a single final JSON."""
    if insurance_type == "life":
        schema = {
            "applicant": {
                "name": "string | null",
                "dateOfBirth": "string | null",
                "fathersName": "string | null",
                "address": "string | null"
            },
            "policyDetails": {
                "policyType": "string | null",
                "coverageAmount": "string | null",
                "planName": "string | null",
                "policyTerms": "string | null",
                "renewalBasis": "string | null"
            },
            "medicalDisclosure": {
                "medicalHistory": "string | null",
                "familyHistory": "string | null",
                "weight": "string | null",
                "height": "string | null",
                "currentMedication": "string | null"
            },
            "lifestyleAssessment": {
                "smokingStatus": "string | null",
                "alcoholConsumption": "string | null",
                "physicalActivity": "string | null",
                "otherHabits": "string | null"
            }
        }
    else:
        schema = {
            "applicant": {
                "name": "string | null",
                "dateOfBirth": "string | null",
                "fathersName": "string | null",
                "address": "string | null"
            },
            "propertyDetails": {
                "propertyAddress": "string | null",
                "propertyType": "string | null",
                "constructionType": "string | null",
                "roofType": "string | null",
                "numberOfStories": "int | null",
                "yearBuilt": "int | null",
                "extras": "object | null"
            },
            "propertyFeatures": {
                "hasCentralAir": "bool | null",
                "hasFireExtinguisher": "bool | null",
                "hasSwimmingPool": "bool | null",
                "extras": "object | null"
            },
            "riskFactors": {
                "distanceToFireStation": "string | null",
                "floodZone": "string | null",
                "hasPreviousClaim": "bool | null",
                "extras": "object | null"
            }
        }

    guidance = (
        "Merge multiple structured summaries of the SAME case into one final JSON. "
        "Rules: Prefer non-null values; when conflicts occur, choose the value that is most specific/complete/consistent across files. "
        "Use reasonable normalization (e.g., ints for years, booleans for yes/no). Leave fields null if truly unknown. "
        "Do NOT include per-file arrays, provenance, comments, or any narrative. Return ONLY the single JSON object."
    )

    payload = {
        "inputs": per_file_summaries
    }

    return (
        f"You consolidate structured insurance data. {guidance}\n\n"
        f"Target schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"<inputs>\n{json.dumps(payload, indent=2)}\n</inputs>\n"
    )


async def consolidate_structured_summaries(
    insurance_type: Literal["life", "property_casualty"],
    per_file_summaries: list[dict]
) -> dict:
    prompt = build_consolidation_prompt(insurance_type, per_file_summaries)
    result = await analyze_with_claude(prompt)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Claude consolidation failed"))
    parsed = extract_json_from_response(result["analysis"])
    return parsed



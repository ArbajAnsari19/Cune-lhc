import os
from datetime import datetime
from botocore.config import Config as BotoConfig
import boto3

NANONETS_API_KEY = os.getenv("NANONETS_API_KEY", "77928432-b113-11f0-843f-0a3a41e3fad1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MAX_WORKERS = 10
OUTPUT_DIR = "outputs"
ANALYSIS_OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

bedrock_config = BotoConfig(
    retries={
        'max_attempts': 10,
        'mode': 'adaptive'
    },
    max_pool_connections=50
)
bedrock_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    config=bedrock_config
)
BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

ANALYSIS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary": {"type": "string"},
        "identified_risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk_description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["Low", "Medium", "High"]},
                    "page_references": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["risk_description", "severity", "page_references"]
            }
        },
        "discrepancies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "discrepancy_description": {"type": "string"},
                    "details": {"type": "string"},
                    "page_references": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["discrepancy_description", "details", "page_references"]
            }
        },
        "medical_timeline": {"type": "string"},
        "property_assessment": {"type": "string"},
        "final_recommendation": {"type": "string"},
        "missing_information": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_description": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["item_description", "notes"]
            }
        },
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": [
        "overall_summary",
        "identified_risks",
        "discrepancies",
        "medical_timeline",
        "property_assessment",
        "final_recommendation",
        "missing_information"
    ]
}

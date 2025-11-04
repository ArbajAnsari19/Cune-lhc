import json
from typing import List, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import bedrock_client, BEDROCK_MODEL_ID, ANALYSIS_OUTPUT_SCHEMA, ANALYSIS_OUTPUT_DIR
import jsonschema

# --- PROMPTS ---
def get_individual_analysis_prompt(insurance_type: str, extracted_data: dict) -> str:
    type_specific_instructions = ""
    if insurance_type == "life":
        type_specific_instructions = """
        Focus on:
        - Medical history and conditions
        - Age, lifestyle factors, and risk behaviors
        - Coverage amounts and beneficiaries
        - Medical timeline if applicable
        - Set 'property_assessment' to "N/A"
        """
    else:
        type_specific_instructions = """
        Focus on:
        - Property details, location, and construction
        - Risk exposures (fire, flood, liability, etc.)
        - Coverage limits and deductibles
        - Property assessment details
        - Set 'medical_timeline' to "N/A"
        """
    consolidated = json.dumps(extracted_data, indent=2)
    return f"""You are an expert insurance underwriter tasked with analyzing extracted document information for {insurance_type.replace('_', ' ').title()} insurance.\n\n{type_specific_instructions}\n\nThe following data was extracted from an insurance document:\n<extracted_data>\n{consolidated}\n</extracted_data>\n\nPlease perform a comprehensive analysis. Your goal is to:\n1. Provide an 'overall_summary' of the document content and its purpose based on the extracted data.\n2. Identify key risks in 'identified_risks'. For each risk, include 'risk_description', 'severity' (Low, Medium, or High), and 'page_references' (list of strings, e.g., ["1", "3-5"], use ["N/A"] if not applicable).\n3. Identify any discrepancies or inconsistencies in 'discrepancies'. For each, include 'discrepancy_description', 'details' (provide specific details of the discrepancy), and 'page_references' (list of strings, e.g., ["2", "10"], use ["N/A"] if not applicable).\n4. Provide a 'medical_timeline' (string, use Markdown for formatting) if the document is medical-related. If not applicable, provide an empty string or "N/A".\n5. Provide a 'property_assessment' (string, use Markdown for formatting) if the document is property-related (e.g., commercial property application). If not applicable, provide an empty string or "N/A".\n6. Formulate a 'final_recommendation' (string, use Markdown for formatting) for the underwriter based on your analysis (e.g., approve, decline with reasons, request more info).\n7. List any critical missing information in 'missing_information'. For each, include 'item_description' and 'notes'.\n8. If you can estimate a 'confidence_score' (0.0 to 1.0) for your overall analysis based on the quality and completeness of the provided extracted data, include it. Otherwise, you can omit it or use a default like 0.75.\n\nStructure your response as a single JSON object matching the following schema precisely. Do not include any explanations or text outside this JSON structure:\n{json.dumps(ANALYSIS_OUTPUT_SCHEMA, indent=2)}\n\nImportant Guidelines:\n- Adhere strictly to the JSON schema provided for the output.\n- If a section like 'identified_risks', 'discrepancies', or 'missing_information' has no items, provide an empty list ([]) for that key.\n- For 'page_references', if the source extracted data does not contain explicit page numbers associated with the information, use ["N/A"].\n- If you can estimate a 'confidence_score' (0.0 to 1.0) for your overall analysis based on the quality and completeness of the provided extracted data, include it. Otherwise, you can omit it or use a default like 0.75.\n\nReturn ONLY the JSON object."""

def get_consolidated_analysis_prompt(insurance_type: str, individual_analyses: List[Dict[str, Any]]) -> str:
    analyses_text = json.dumps(individual_analyses, indent=2)
    return f"""You are a senior insurance underwriter conducting a comprehensive portfolio review for {insurance_type.replace('_', ' ').title()} insurance applications.\n\nYou have been provided with individual analyses from multiple documents. Your task is to create a consolidated final analysis that synthesizes all findings.\n<individual_analyses>\n{analyses_text}\n</individual_analyses>\n\nPlease provide a comprehensive consolidated analysis with the following:\n\n1. **Executive Summary**: A high-level overview of all documents analyzed, key patterns, and overall assessment (2-3 paragraphs).\n2. **Aggregated Risk Profile**: \n   - Categorize and summarize all identified risks across documents\n   - Highlight the most severe risks\n   - Identify any patterns or recurring risk factors\n   - Provide risk distribution statistics (e.g., X High, Y Medium, Z Low severity risks)\n3. **Critical Discrepancies & Concerns**:\n   - List all discrepancies found across documents\n   - Highlight any cross-document inconsistencies\n   - Flag items requiring immediate attention\n4. **Consolidated Recommendations**:\n   - Overall underwriting decision (Approve/Decline/Request Additional Information)\n   - Specific conditions or exclusions if approval is recommended\n   - Priority list of missing information needed from all documents\n   - Suggested premium adjustments or special terms if applicable\n5. **Portfolio-Level Insights**:\n   - Overall confidence level in the application package\n   - Any red flags that span multiple documents\n   - Opportunities for risk mitigation\n   - Next steps for the underwriting team\n6. **Document-Specific Summary Table**: \n   A brief table or list summarizing each document's:\n   - Document name/number\n   - Primary purpose\n   - Key finding\n   - Individual recommendation\n\nFormat your response in clear Markdown with appropriate headers, bullet points, and tables where helpful. Be concise but thorough, prioritizing actionable insights for the underwriting decision-maker.\n\nYour consolidated analysis:"""

async def analyze_with_claude(prompt: str, max_tokens: int = 4096) -> dict:
    try:
        loop = asyncio.get_event_loop()
        def call_claude():
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            if 'content' in response_body and len(response_body['content']) > 0:
                return response_body['content'][0]['text']
            else:
                raise ValueError("Unexpected response format from Bedrock")
        with ThreadPoolExecutor(max_workers=1) as executor:
            response_text = await loop.run_in_executor(executor, call_claude)
        return {"success": True, "analysis": response_text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_json_from_response(response_text: str) -> dict:
    import re
    response_text = response_text.strip()
    code_block_pattern = r'```(?:json)?\s*(.+?)\s*```'
    code_match = re.search(code_block_pattern, response_text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
    else:
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            json_str = response_text[start_idx:end_idx+1]
        else:
            json_str = response_text
    return json.loads(json_str)

def validate_analysis_schema(analysis_json: dict) -> Tuple[bool, str]:
    try:
        jsonschema.validate(instance=analysis_json, schema=ANALYSIS_OUTPUT_SCHEMA)
        return True, ""
    except jsonschema.ValidationError as e:
        error_msg = getattr(e, 'message', str(e))
        error_path = ' -> '.join(str(p) for p in e.path) if e.path else 'root'
        return False, f"Schema validation failed: {error_msg} (Path: {error_path})"
    except Exception as e:
        required_fields = ANALYSIS_OUTPUT_SCHEMA.get("required", [])
        missing_fields = [field for field in required_fields if field not in analysis_json]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        if "identified_risks" in analysis_json and not isinstance(analysis_json["identified_risks"], list):
            return False, "Field 'identified_risks' must be a list"
        if "discrepancies" in analysis_json and not isinstance(analysis_json["discrepancies"], list):
            return False, "Field 'discrepancies' must be a list"
        if "missing_information" in analysis_json and not isinstance(analysis_json["missing_information"], list):
            return False, "Field 'missing_information' must be a list"
        return True, ""

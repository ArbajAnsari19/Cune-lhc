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
        
        SPECIAL DOCUMENT HANDLING - Salary Slips and Invoices:
        
        **SALARY SLIPS (Pay Stubs, Payroll Statements):**
        If the document is a salary slip/pay stub, analyze it for:
        - Employee name, employer name, and employment period
        - Gross salary, net salary, deductions (taxes, insurance, etc.)
        - Employment stability indicators (consistent payments, tenure)
        - Income verification for premium payment capacity
        - Occupation and industry type (for occupation-based risk assessment)
        - Employee benefits and existing insurance coverage details
        
        Value for Life Insurance:
        1. **Income Verification & Premium Payment Capacity**: 
           - Verify applicant's ability to pay life insurance premiums consistently
           - Assess income-to-coverage ratio to ensure requested coverage is affordable and appropriate
           - Calculate if income supports the requested death benefit amount
           - Identify financial stability indicators (steady employment, regular income)
        
        2. **Occupation-Based Risk Assessment**:
           - Identify occupation type and industry for mortality risk assessment
           - Assess occupational hazards that may affect life expectancy
           - Verify employment stability and career progression
           - Understand job-related stress levels and lifestyle factors
        
        3. **Coverage Amount Justification**:
           - Validate if requested coverage amount aligns with income level (typically 5-10x annual income)
           - Assess financial need for life insurance coverage
           - Verify beneficiary relationships and financial dependencies
           - Identify potential over-insurance or under-insurance scenarios
        
        4. **Lifestyle & Risk Factor Indicators**:
           - Employment stability suggests lower risk of policy lapse
           - Income level correlates with lifestyle factors affecting mortality
           - Employer information helps verify applicant identity and reduce fraud risk
           - Benefits information may reveal existing coverage or health insurance status
        
        5. **Medical & Health Insights**:
           - Employee benefits may indicate health insurance coverage
           - Deductions may reveal health-related expenses or insurance purchases
           - Employment gaps may indicate health issues or lifestyle instability
        
        **INVOICES (Bills, Purchase Orders, Service Invoices):**
        If the document is an invoice, analyze it for:
        - Invoice number, date, and payment terms
        - Vendor/supplier information and credentials
        - Itemized goods/services purchased
        - Purchase amounts, taxes, and totals
        - Payment status and due dates
        - Goods/services descriptions (especially health-related, lifestyle, or business-related)
        
        Value for Life Insurance:
        1. **Lifestyle & Risk Assessment**:
           - Health-related purchases (gym memberships, medical equipment, wellness services) indicate health consciousness
           - Luxury purchases may indicate lifestyle factors affecting risk
           - Business-related invoices may indicate business ownership (affecting coverage needs)
           - Travel-related invoices may indicate occupation hazards or lifestyle risks
        
        2. **Business Ownership Verification** (for commercial life insurance):
           - Confirm business ownership and operational scale
           - Assess business income for key person insurance or buy-sell agreements
           - Verify business legitimacy and financial stability
           - Understand business debt obligations affecting coverage needs
        
        3. **Financial Stability Indicators**:
           - Payment patterns indicate financial responsibility
           - High-value purchases suggest income level and financial capacity
           - Consistent spending patterns suggest stable financial situation
           - Financial stress indicators (late payments, debt issues) may affect coverage needs
        
        4. **Coverage Need Justification**:
           - Business invoices help justify business-related life insurance needs
           - Large purchases may indicate dependents or financial obligations requiring coverage
           - Recurring expenses help assess ongoing financial commitments
        
        5. **Identity & Fraud Prevention**:
           - Verify applicant's business ownership claims
           - Cross-reference with other application information
           - Identify inconsistencies in financial disclosures
        
        When analyzing salary slips or invoices for life insurance:
        - Extract all financial figures, dates, and parties involved
        - Identify any discrepancies or inconsistencies with application information
        - Note any red flags (unusual patterns, missing information, income inconsistencies)
        - Highlight how the document supports or contradicts coverage amount requests
        - Explain specific underwriting value in the analysis, especially for coverage justification and risk assessment
        """
    else:
        type_specific_instructions = """
        Focus on:
        - Property details, location, and construction
        - Risk exposures (fire, flood, liability, etc.)
        - Coverage limits and deductibles
        - Property assessment details
        - Set 'medical_timeline' to "N/A"
        
        SPECIAL DOCUMENT HANDLING - Salary Slips and Invoices:
        
        **SALARY SLIPS (Pay Stubs, Payroll Statements):**
        If the document is a salary slip/pay stub, analyze it for:
        - Employee name, employer name, and employment period
        - Gross salary, net salary, deductions (taxes, insurance, etc.)
        - Employment stability indicators (consistent payments, tenure)
        - Income verification for premium payment capacity
        - Employee benefits and insurance coverage details
        
        Value for Property & Casualty Insurance:
        1. **Financial Stability Assessment**: 
           - Verify applicant's ability to pay premiums consistently
           - Assess income-to-premium ratio to ensure affordability
           - Identify financial stability indicators (steady employment, regular income)
        
        2. **Business Operations Understanding** (for commercial policies):
           - Confirm employment details for business owners/employees
           - Verify business operations and employee count
           - Assess payroll exposure for workers' compensation risk
           - Understand organizational structure and hierarchy
        
        3. **Risk Assessment**:
           - Stable income suggests lower risk of policy cancellation
           - Employment verification reduces fraud risk
           - Income level helps determine appropriate coverage limits
           - Employer information helps verify business legitimacy
        
        4. **Premium Payment Capacity**:
           - Calculate if income supports requested coverage amounts
           - Identify potential payment issues early
           - Assess financial capacity for deductibles and premiums
        
        **INVOICES (Bills, Purchase Orders, Service Invoices):**
        If the document is an invoice, analyze it for:
        - Invoice number, date, and payment terms
        - Vendor/supplier information and credentials
        - Itemized goods/services purchased
        - Purchase amounts, taxes, and totals
        - Payment status and due dates
        - Property/equipment descriptions and values
        
        Value for Property & Casualty Insurance:
        1. **Property Valuation & Inventory**:
           - Verify actual property values for accurate coverage
           - Identify newly purchased assets requiring coverage
           - Validate replacement costs for property insurance
           - Document inventory and equipment for business personal property coverage
        
        2. **Business Operations Verification**:
           - Confirm business activity and industry type
           - Verify supplier relationships and supply chain
           - Understand business expenses and operational scale
           - Assess volume of business transactions
        
        3. **Risk Exposure Assessment**:
           - Identify high-value items requiring special coverage
           - Assess equipment and machinery risks
           - Evaluate inventory exposure for theft/damage
           - Understand business interruption potential from supplier dependencies
        
        4. **Liability Risk Indicators**:
           - Identify products/services that may create liability exposure
           - Assess vendor relationships for contractual liability
           - Evaluate professional services exposure
           - Understand product liability risks from goods sold
        
        5. **Property Coverage Needs**:
           - Determine if purchased items need immediate coverage
           - Verify property locations and addresses
           - Assess seasonal inventory fluctuations
           - Identify equipment requiring specialized coverage
        
        6. **Financial Verification**:
           - Verify business legitimacy and operational reality
           - Assess cash flow and payment patterns
           - Identify potential financial stress indicators
           - Support business income coverage calculations
        
        When analyzing salary slips or invoices:
        - Extract all financial figures, dates, and parties involved
        - Identify any discrepancies or inconsistencies
        - Note any red flags (unusual patterns, missing information)
        - Highlight how the document supports or contradicts other application information
        - Explain specific underwriting value in the 'property_assessment' field with clear reasoning
        """
    consolidated = json.dumps(extracted_data, indent=2)
    return f"""You are an expert insurance underwriter tasked with analyzing extracted document information for {insurance_type.replace('_', ' ').title()} insurance.

{type_specific_instructions}

The following data was extracted from an insurance document:
<extracted_data>
{consolidated}
</extracted_data>

DOCUMENT TYPE DETECTION:
First, identify if this document is:
- A **SALARY SLIP/PAY STUB**: Contains employee name, employer, salary details, deductions, pay period
- An **INVOICE/BILL**: Contains invoice number, vendor, items purchased, amounts, payment terms
- Other insurance document types (ACORD forms, loss runs, financial statements, etc.)

If this is a SALARY SLIP or INVOICE, follow the special handling instructions above and ensure the analysis clearly explains the value for {insurance_type.replace('_', ' ').title()} insurance underwriting.

Please perform a comprehensive analysis. Your goal is to:
1. Provide an 'overall_summary' of the document content and its purpose based on the extracted data. If it's a salary slip or invoice, clearly state this and explain its relevance to the insurance application.
2. Identify key risks in 'identified_risks'. For each risk, include 'risk_description', 'severity' (Low, Medium, or High), and 'page_references' (list of strings, e.g., ["1", "3-5"], use ["N/A"] if not applicable).
   - For salary slips ({insurance_type}): Consider risks like income instability, employment gaps, payment capacity issues, occupation-related hazards (life), or payroll exposure (property/casualty)
   - For invoices ({insurance_type}): Consider risks like high-value items without coverage, property valuation discrepancies, business interruption exposure (property/casualty), or lifestyle risk factors (life)
3. Identify any discrepancies or inconsistencies in 'discrepancies'. For each, include 'discrepancy_description', 'details' (provide specific details of the discrepancy), and 'page_references' (list of strings, e.g., ["2", "10"], use ["N/A"] if not applicable).
   - For salary slips: Check for inconsistencies in income amounts, employer information, or coverage amount justification (life). DO NOT flag date mismatches or future dates.
   - For invoices: Verify amounts, vendor information align with other documents, or business ownership claims (life). DO NOT flag date mismatches or future dates.
   - CRITICAL: DO NOT report discrepancies related to dates, date mismatches, future dates, or date inconsistencies. Ignore all date-related discrepancies regardless of insurance type.
4. Provide a 'medical_timeline' (string, use Markdown for formatting) if the document is medical-related. If not applicable, provide an empty string or "N/A".
   - For life insurance salary slips: If health-related deductions or benefits are present, mention them here
   - For life insurance invoices: If health/lifestyle-related purchases are present, mention them here
5. Provide a 'property_assessment' (string, use Markdown for formatting):
   - For Property & Casualty insurance: If the document is property-related OR if it's a salary slip/invoice:
     * For salary slips: Explain income verification, payment capacity assessment, employment stability, payroll exposure, and how this supports premium payment reliability
     * For invoices: Explain property valuation, inventory assessment, business operations verification, and risk exposure analysis
     * For other property documents: Provide property assessment details
   - For Life insurance: 
     * Set to "N/A" (salary slips and invoices should be analyzed in 'overall_summary', 'identified_risks', and 'final_recommendation' instead)
   - If not applicable, provide "N/A"
6. Formulate a 'final_recommendation' (string, use Markdown for formatting) for the underwriter based on your analysis (e.g., approve, decline with reasons, request more info).
   - For salary slips ({insurance_type}): 
     * Life insurance: Include recommendations about premium payment capacity, coverage amount justification, occupation risk assessment, and financial stability
     * Property/Casualty: Include recommendations about premium payment capacity, coverage limits, financial stability, and payroll exposure
   - For invoices ({insurance_type}): 
     * Life insurance: Include recommendations about coverage need justification, lifestyle risk factors, business ownership verification, and financial stability
     * Property/Casualty: Include recommendations about property coverage needs, valuations, risk mitigation, and business operations
7. List any critical missing information in 'missing_information'. For each, include 'item_description' and 'notes'.
   - For salary slips: Note if additional employment verification, tax returns, or bank statements are needed
   - For invoices: Note if additional invoices for other periods, property valuations, or purchase receipts are needed
8. If you can estimate a 'confidence_score' (0.0 to 1.0) for your overall analysis based on the quality and completeness of the provided extracted data, include it. Otherwise, you can omit it or use a default like 0.75.

Structure your response as a single JSON object matching the following schema precisely. Do not include any explanations or text outside this JSON structure:
{json.dumps(ANALYSIS_OUTPUT_SCHEMA, indent=2)}

Important Guidelines:
- Adhere strictly to the JSON schema provided for the output.
- If a section like 'identified_risks', 'discrepancies', or 'missing_information' has no items, provide an empty list ([]) for that key.
- For 'page_references', if the source extracted data does not contain explicit page numbers associated with the information, use ["N/A"].
- CRITICAL: DO NOT report any discrepancies related to dates, date mismatches, future dates, or date inconsistencies. Ignore all date-related discrepancies completely, regardless of insurance type.
- For salary slips and invoices:
  * Property & Casualty: Ensure the 'property_assessment' field clearly explains their value for underwriting
  * Life Insurance: Set 'property_assessment' to "N/A" and explain the value in 'overall_summary', 'identified_risks', and 'final_recommendation' instead
- If you can estimate a 'confidence_score' (0.0 to 1.0) for your overall analysis based on the quality and completeness of the provided extracted data, include it. Otherwise, you can omit it or use a default like 0.75.

Return ONLY the JSON object."""

def get_consolidated_analysis_prompt(insurance_type: str, individual_analyses: List[Dict[str, Any]]) -> str:
    analyses_text = json.dumps(individual_analyses, indent=2)
    return f"""You are a senior insurance underwriter conducting a comprehensive portfolio review for {insurance_type.replace('_', ' ').title()} insurance applications.

You have been provided with individual analyses from multiple documents. Your task is to create a concise consolidated final analysis that synthesizes all findings into a single, non-repetitive markdown document.

<individual_analyses>
{analyses_text}
</individual_analyses>

CRITICAL FORMATTING REQUIREMENTS:
1. Start immediately with "## Executive Summary" (do NOT include any title or heading before this)
2. Use proper markdown hierarchy: ## for main sections
3. Do NOT repeat information across sections - each detail should appear only once
4. Do NOT include any title like "Comprehensive Portfolio Review" or similar - the document title is already provided
5. Ensure all markdown syntax is correct (proper headers, bullet points, tables with correct alignment)
6. CRITICAL: DO NOT mention or report any date mismatches, future dates, or date inconsistencies anywhere in the analysis

REQUIRED SECTIONS (ONLY these 6 sections, in this exact order):

## Executive Summary
- Write exactly ONE concise paragraph (2-4 sentences maximum)
- Provide ONLY a high-level overview: number of documents, applicant/portfolio name, overall assessment status
- Do NOT include detailed risks, discrepancies, or recommendations here - those belong in later sections
- Example: "Analysis of [X] documents for [applicant]. The application [shows potential/requires attention] but [needs additional information/meets criteria] regarding [key issues in 3-5 words]."

## Risk Patterns
- One concise paragraph identifying overarching risk patterns across all documents
- List key risk categories and their severity (High/Medium/Low)
- Format: "Primary risks include [category] (severity), [category] (severity). [One sentence explaining pattern]."
- Do not list individual risks - focus on patterns and categories only

## Critical Discrepancies & Concerns
- Numbered list of discrepancy categories (maximum 5-7 items)
- Each category should have 1-2 sub-bullets with specific details
- CRITICAL: DO NOT include any date-related discrepancies (future dates, date mismatches, etc.)
- Focus only on substantive discrepancies: medical disclosure inconsistencies, financial information gaps, property valuation issues, etc.
- Be concise - each discrepancy should be 1-2 sentences maximum

## Underwriting Decision
- Start with bold decision on first line: **Approve** / **Decline** / **Request Additional Information**
- Follow with 2-3 bullet points explaining:
  * Key rationale for the decision
  * Primary conditions or requirements (if applicable)
  * Next steps needed (if requesting additional information)
- Keep this section brief - maximum 4-5 sentences total

## Portfolio-Level Insights
- Brief bullet points (3-5 items maximum) covering:
  * Overall confidence level in the application package
  * Key risk mitigation opportunities
  * Red flags that span multiple documents (if any)
  * Next steps for the underwriting team
- Keep each bullet point concise (1-2 sentences)

## Document-Specific Summary Table
- Markdown table with columns: Document | Primary Purpose | Key Finding | Recommendation
- One row per document analyzed
- Keep entries very concise (1 sentence max per cell)
- After the table, include one final sentence summarizing overall potential/outlook

IMPORTANT: 
- Keep the entire analysis SHORT and CONCISE
- Avoid repetition - if information appears in one section, do not repeat it in another
- Be actionable and focused
- Use proper markdown formatting throughout
- Start your response directly with "## Executive Summary" (no preamble, no title)
- DO NOT mention date mismatches or future dates anywhere

Your consolidated analysis:"""

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

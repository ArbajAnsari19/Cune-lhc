from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Literal, Optional
from utils.s3_service import s3_service
from datetime import datetime
import asyncio
import os
import json
import shutil
import tempfile

from services.extract import process_pdf_async, poll_until_ready, get_latest_json_files
from services.analyze import (
    get_individual_analysis_prompt,
    get_consolidated_analysis_prompt,
    analyze_with_claude,
    extract_json_from_response,
    validate_analysis_schema
)
from services.doc_classification import classify_verification_documents
from services.user_kyc import generate_user_kyc
from services.structured_summary import run_structured_summary_prompt, consolidate_structured_summaries
from services.chat import answer_query
from models.schemas import LifeSummary, PropertyCasualtySummary
from config.settings import (
    NANONETS_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    OUTPUT_DIR,
    ANALYSIS_OUTPUT_DIR
)

router = APIRouter()

@router.get("/")
async def root():
    return {
        "message": "PDF Extraction & Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "extraction": ["/extract"],
            "analysis": ["/analysis"],
            "kyc": ["/get_kyc"],
            "chat": ["/chat"]
        }
    }

@router.get("/health")
async def health():
    json_files = get_latest_json_files()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "json_files_available": len(json_files),
        "aws_bedrock_configured": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    }

@router.post("/extract")
async def extract_pdfs(
    files: List[UploadFile] = File(...),
    submission_id: Optional[str] = Query(None, description="Submission ID for organizing files in S3")
):
    """
    Extract PDFs to JSON and optionally upload to S3
    
    Args:
        files: List of PDF files to extract
        submission_id: Optional submission ID for organizing files in S3
    
    Returns:
        JSON response with extraction results
    """
    # If submission_id provided, clear only that submission's directory; otherwise clear all
    if submission_id:
        # Clear only submission-specific directory
        submission_output_dir = os.path.join(OUTPUT_DIR, submission_id)
        if os.path.exists(submission_output_dir):
            for filename in os.listdir(submission_output_dir):
                file_path = os.path.join(submission_output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    else:
        # Clear all outputs (backward compatibility)
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Clear analysis output directory for this submission (if ID provided)
    if submission_id:
        submission_analysis_dir = os.path.join(ANALYSIS_OUTPUT_DIR, submission_id)
        if os.path.exists(submission_analysis_dir):
            for filename in os.listdir(submission_analysis_dir):
                file_path = os.path.join(submission_analysis_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No files provided. Upload at least one PDF in the 'files' field."
        )
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}. Only PDF files are allowed.")
    temp_dir = tempfile.mkdtemp()
    try:
        # Pass submission_id to process_pdf_async for S3 upload
        tasks = [process_pdf_async(f, NANONETS_API_KEY, temp_dir, submission_id, upload_to_s3=True) for f in files]
        results = await asyncio.gather(*tasks)

        response_data = {
            "total_files": len(files),
            "results": results,
            "success_count": sum(1 for r in results if r.get("success")),
            "failure_count": sum(1 for r in results if not r.get("success")),
            "timestamp": datetime.now().isoformat()
        }
        return JSONResponse(content=response_data, status_code=200)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/get_kyc")
async def get_kyc(
    submission_id: Optional[str] = Query(None, description="Optional submission ID for organizing files per submission")
):
    """
    Run document classification over outputs and then build user_kyc.json.
    
    If submission_id is provided:
    - Reads JSON files from outputs/{submission_id}/
    - Moves classified files to verification_documents/{submission_id}/
    - Generates user_kyc.json in verification_documents/{submission_id}/
    
    If submission_id is not provided (backward compatible):
    - Reads JSON files from outputs/
    - Moves classified files to verification_documents/
    - Generates user_kyc.json in verification_documents/
    """
    classification_status = "success"
    kyc_status = "success"
    
    # Determine verification directory based on submission_id
    base_root = os.path.abspath(os.path.join(OUTPUT_DIR, os.pardir))
    if submission_id:
        verification_dir = os.path.join(base_root, "verification_documents", submission_id)
    else:
        verification_dir = os.path.join(base_root, "verification_documents")
    
    kyc_file = os.path.join(verification_dir, "user_kyc.json")

    try:
        classify_verification_documents(submission_id=submission_id)
    except Exception as e:
        classification_status = f"error: {str(e)}"

    try:
        generate_user_kyc(submission_id=submission_id)
    except Exception as e:
        kyc_status = f"error: {str(e)}"

    return JSONResponse(content={
        "submission_id": submission_id,
        "classification": classification_status,
        "kyc": kyc_status,
        "verification_dir": verification_dir,
        "kyc_file": kyc_file,
        "timestamp": datetime.now().isoformat()
    }, status_code=200)

@router.post("/analysis")
async def analyze_documents(
    insurance_type: Literal["life", "property_casualty"] = Query(..., description="Type of insurance analysis"),
    submission_id: Optional[str] = Query(None, description="Submission ID for organizing files in S3")
):
    """
    Analyze extracted documents and optionally upload to S3
    
    Args:
        insurance_type: Type of insurance analysis ('life' or 'property_casualty')
        submission_id: Optional submission ID for organizing files in S3
    
    Returns:
        JSON response with analysis results
    """
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise HTTPException(
            status_code=500,
            detail="AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    # Get JSON files from S3 if submission_id provided, otherwise from local filesystem
    json_files = get_latest_json_files(submission_id=submission_id, from_s3=(submission_id is not None))
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail="No JSON files found. Please extract PDFs first."
        )
    
    individual_analyses = []
    analysis_results = []
    
    # Create submission-specific analysis directory if needed
    analysis_output_dir = ANALYSIS_OUTPUT_DIR
    if submission_id:
        analysis_output_dir = os.path.join(ANALYSIS_OUTPUT_DIR, submission_id)
        os.makedirs(analysis_output_dir, exist_ok=True)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                extracted_data = f.read()
            extracted_data = json.loads(extracted_data)
            prompt = get_individual_analysis_prompt(insurance_type, extracted_data)
            result = await analyze_with_claude(prompt)
            if result["success"]:
                try:
                    analysis_json = extract_json_from_response(result["analysis"])
                    is_valid, validation_error = validate_analysis_schema(analysis_json)
                    if not is_valid:
                        analysis_results.append({
                            "file": os.path.basename(json_file),
                            "status": "error",
                            "error": f"Schema validation failed: {validation_error}",
                            "raw_response": result["analysis"][:500]
                        })
                        continue
                    analysis_with_metadata = {
                        "source_file": os.path.basename(json_file),
                        "analysis_timestamp": datetime.now().isoformat(),
                        "insurance_type": insurance_type,
                        "analysis": analysis_json
                    }
                    individual_analyses.append(analysis_json)
                    base_name = os.path.splitext(os.path.basename(json_file))[0]
                    
                    # Save locally first
                    analysis_file = os.path.join(
                        analysis_output_dir,
                        f"{base_name}_analysis_{insurance_type}.json"
                    )
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_with_metadata, f, ensure_ascii=False, indent=2)
                    
                    # Upload to S3 if submission_id provided
                    s3_url = None
                    s3_key = None
                    if submission_id:
                        s3_key = f"lnh-submissions/{submission_id}/analysis/{base_name}_analysis_{insurance_type}.json"
                        s3_url = s3_service.upload_file(analysis_file, s3_key, content_type="application/json")
                    
                    analysis_results.append({
                        "file": os.path.basename(json_file),
                        "status": "success",
                        "analysis": analysis_json,  # Include the actual analysis JSON data
                        "analysis_saved_to": analysis_file,
                        "s3_url": s3_url,
                        "s3_key": s3_key
                    })
                except Exception as e:
                    analysis_results.append({
                        "file": os.path.basename(json_file),
                        "status": "error",
                        "error": f"Failed to parse Claude response as JSON: {str(e)}",
                        "raw_response": result["analysis"][:500]
                    })
            else:
                analysis_results.append({
                    "file": os.path.basename(json_file),
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                })
        except Exception as e:
            analysis_results.append({
                "file": os.path.basename(json_file),
                "status": "error",
                "error": str(e)
            })
    consolidated_analysis = None
    consolidated_s3_url = None
    if individual_analyses:
        try:
            consolidated_prompt = get_consolidated_analysis_prompt(insurance_type, individual_analyses)
            consolidated_result = await analyze_with_claude(consolidated_prompt, max_tokens=8192)
            if consolidated_result["success"]:
                consolidated_analysis = consolidated_result["analysis"]
                
                # Save locally first
                consolidated_file = os.path.join(
                    analysis_output_dir,
                    f"consolidated_analysis_{insurance_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                )
                with open(consolidated_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Consolidated {insurance_type.replace('_', ' ').title()} Insurance Analysis\n\n")
                    f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
                    f.write(f"**Documents Analyzed:** {len(individual_analyses)}\n\n")
                    f.write("---\n\n")
                    f.write(consolidated_analysis)
                
                # Upload to S3 if submission_id provided
                if submission_id:
                    s3_key = f"lnh-submissions/{submission_id}/analysis/consolidated_analysis_{insurance_type}.md"
                    consolidated_s3_url = s3_service.upload_file(consolidated_file, s3_key, content_type="text/markdown")
        except Exception as e:
            consolidated_analysis = f"Error generating consolidated analysis: {str(e)}"
    
    response_data = {
        "insurance_type": insurance_type,
        "total_files_processed": len(json_files),
        "successful_analyses": len(individual_analyses),
        "failed_analyses": len(json_files) - len(individual_analyses),
        "individual_results": analysis_results,
        "consolidated_analysis": consolidated_analysis,
        "consolidated_s3_url": consolidated_s3_url,
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(content=response_data, status_code=200)


@router.post("/structured_summary")
async def structured_summary(
    insurance_type: Literal["life", "property_casualty"] = Query(..., description="Choose 'life' or 'property_casualty'"),
    submission_id: Optional[str] = Query(None, description="Submission ID for organizing files in S3")
):
    """
    Generate structured summary from extracted documents
    
    Args:
        insurance_type: Type of insurance ('life' or 'property_casualty')
        submission_id: Optional submission ID for organizing files in S3
    
    Returns:
        JSON response with structured summary
    """
    # Get JSON files from S3 if submission_id provided, otherwise from local filesystem
    json_files = get_latest_json_files(submission_id=submission_id, from_s3=(submission_id is not None))
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail="No JSON files found in outputs directory. Please extract PDFs first."
        )

    summaries = []
    errors = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                extracted = json.load(f)
            
            # Log extracted data structure for debugging
            print(f"📄 Processing file: {os.path.basename(json_file)}")
            print(f"📊 Extracted data keys: {list(extracted.keys()) if isinstance(extracted, dict) else 'Not a dict'}")
            if isinstance(extracted, dict):
                # Log sample of extracted data (first 500 chars)
                extracted_str = json.dumps(extracted, indent=2)[:500]
                print(f"📋 Sample extracted data: {extracted_str}...")
            
            parsed = await run_structured_summary_prompt(insurance_type, extracted)
            
            # Log what Claude returned
            print(f"🤖 Claude parsed response keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
            parsed_str = json.dumps(parsed, indent=2)[:500]
            print(f"📋 Sample parsed data: {parsed_str}...")

            # Validate/normalize against Pydantic schemas for consistent output
            if insurance_type == "life":
                model = LifeSummary(**parsed)
            else:
                model = PropertyCasualtySummary(**parsed)

            summaries.append({
                "source_file": os.path.basename(json_file),
                "summary": model.model_dump()
            })
        except Exception as e:
            errors.append({
                "source_file": os.path.basename(json_file),
                "error": str(e)
            })

    # If there are parsed summaries, consolidate them into a single final JSON
    try:
        consolidated_input = [s["summary"] for s in summaries]
        print(f"🔄 Consolidating {len(consolidated_input)} summaries...")
        for i, summary in enumerate(consolidated_input):
            print(f"📄 Summary {i+1} keys: {list(summary.keys()) if isinstance(summary, dict) else 'Not a dict'}")
            summary_str = json.dumps(summary, indent=2)[:500]
            print(f"📋 Sample summary {i+1}: {summary_str}...")
        
        final_summary = await consolidate_structured_summaries(insurance_type, consolidated_input)
        
        print(f"✅ Consolidated summary keys: {list(final_summary.keys()) if isinstance(final_summary, dict) else 'Not a dict'}")
        final_str = json.dumps(final_summary, indent=2)[:500]
        print(f"📋 Sample consolidated summary: {final_str}...")
        
        # Upload to S3 if submission_id provided
        if submission_id and final_summary:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(final_summary, tmp_file, ensure_ascii=False, indent=2)
                tmp_path = tmp_file.name
            
            s3_key = f"lnh-submissions/{submission_id}/summary/structured_summary_{insurance_type}.json"
            s3_url = s3_service.upload_file(tmp_path, s3_key, content_type="application/json")
            os.unlink(tmp_path)  # Clean up temp file
            
            # Add S3 URL to response
            if isinstance(final_summary, dict):
                final_summary["s3_url"] = s3_url
                final_summary["s3_key"] = s3_key
        
        # Return only the consolidated JSON as requested
        return JSONResponse(content=final_summary, status_code=200)
    except Exception as e:
        # If consolidation fails, still return diagnostics to client
        return JSONResponse(content={
            "insurance_type": insurance_type,
            "total_files": len(json_files),
            "summaries": summaries,
            "errors": errors + [{"source_file": "CONSOLIDATION", "error": str(e)}],
            "timestamp": datetime.now().isoformat()
        }, status_code=200)


@router.post("/chat")
async def chat_with_documents(
    query: str = Query(..., description="The question to ask about the documents"),
    submission_id: str = Query(..., description="Submission ID to identify which documents to query"),
    insurance_type: Literal["life", "property_casualty"] = Query("property_casualty", description="Type of insurance analysis"),
    top_k: int = Query(6, ge=1, le=20, description="Number of relevant chunks to retrieve"),
    temperature: float = Query(0.2, ge=0.0, le=2.0, description="Temperature for response generation"),
    from_s3: bool = Query(True, description="Whether to read JSON files from S3 (True) or local filesystem (False)")
):
    """
    Chat with documents using RAG (Retrieval-Augmented Generation)
    
    This endpoint allows you to ask questions about extracted documents for a specific submission.
    It automatically initializes embeddings when JSON files are available in outputs/{submission_id}/ or S3.
    
    Args:
        query: The question to ask about the documents
        submission_id: Submission ID to identify which documents to query (must match the submission_id used in /extract)
        insurance_type: Type of insurance analysis ('life' or 'property_casualty')
        top_k: Number of relevant chunks to retrieve (1-20)
        temperature: Temperature for response generation (0.0-2.0)
        from_s3: Whether to read JSON files from S3 (default True for production) or local filesystem
    
    Returns:
        JSON response with the answer
    """
    try:
        answer = await answer_query(
            submission_id=submission_id,
            query=query,
            insurance_type=insurance_type,
            top_k=top_k,
            temperature=temperature,
            from_s3=from_s3
        )
        
        return JSONResponse(content={
            "submission_id": submission_id,
            "query": query,
            "answer": answer,
            "insurance_type": insurance_type,
            "timestamp": datetime.now().isoformat()
        }, status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat query: {str(e)}"
        )

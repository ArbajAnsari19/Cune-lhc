import requests
import json
import os
import time
import tempfile
import shutil
from typing import List, Optional
from fastapi import UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import OUTPUT_DIR

def poll_until_ready(record_id: str, api_key: str, max_wait: int = 120, interval: int = 10):
    url = f"https://extraction-api.nanonets.com/files/{record_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    start_time = time.time()
    while True:
        response = requests.get(url, headers=headers, timeout=60)
        data = response.json()
        content = data.get("content", "")
        if content and not data.get("processing_status") == "processing":
            try:
                return json.loads(content)
            except Exception:
                return content
        if time.time() - start_time > max_wait:
            raise TimeoutError(f"Polling timed out after {max_wait} seconds for record_id {record_id}")
        time.sleep(interval)

def extract_pdf_to_json_sync(pdf_path: str, api_key: str, submission_id: Optional[str] = None, upload_to_s3: bool = True) -> dict:
    """
    Extract PDF to JSON and optionally upload to S3
    
    Args:
        pdf_path: Path to PDF file
        api_key: Nanonets API key
        submission_id: Optional submission ID for S3 path organization
        upload_to_s3: Whether to upload to S3 after extraction
    
    Returns:
        Dictionary with success status, content, filename, and S3 URL
    """
    from utils.s3_service import s3_service
    import tempfile
    
    extract_url = "https://extraction-api.nanonets.com/extract"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"output_type": "flat-json"}
    try:
        with open(pdf_path, "rb") as file:
            files = {"file": file}
            response = requests.post(extract_url, headers=headers, files=files, data=data, timeout=90)
        response.raise_for_status()
        response_data = response.json()
        content_str = response_data.get("content")
        record_id = response_data.get("record_id")
        if content_str:
            content_json = json.loads(content_str)
        elif record_id:
            content_json = poll_until_ready(record_id, api_key)
        else:
            return {"success": False, "error": "No record_id or content returned", "response": response_data}
        
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        filename = os.path.basename(pdf_path)
        
        # Save locally first (for backward compatibility)
        local_output_dir = OUTPUT_DIR
        if submission_id:
            local_output_dir = os.path.join(OUTPUT_DIR, submission_id)
            os.makedirs(local_output_dir, exist_ok=True)
        
        local_json_path = os.path.join(local_output_dir, f"{base_name}.json")
        with open(local_json_path, "w", encoding="utf-8") as f:
            json.dump(content_json, f, ensure_ascii=False, indent=2)
        
        result = {
            "success": True,
            "content": content_json,
            "filename": filename,
            "saved_to": local_json_path
        }
        
        # Upload to S3 if requested
        if upload_to_s3 and submission_id:
            s3_key = f"lnh-submissions/{submission_id}/outputs/{base_name}.json"
            s3_url = s3_service.upload_file(local_json_path, s3_key, content_type="application/json")
            result["s3_url"] = s3_url
            result["s3_key"] = s3_key
        
        return result
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}", "filename": os.path.basename(pdf_path)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}", "filename": os.path.basename(pdf_path)}

async def process_pdf_async(pdf_file: UploadFile, api_key: str, temp_dir: str, submission_id: Optional[str] = None, upload_to_s3: bool = True) -> dict:
    temp_path = os.path.join(temp_dir, pdf_file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, extract_pdf_to_json_sync, temp_path, api_key, submission_id, upload_to_s3)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_latest_json_files(submission_id: Optional[str] = None, from_s3: bool = False) -> List[str]:
    """
    Get JSON files, either from local filesystem or S3
    
    Args:
        submission_id: Optional submission ID to filter files
        from_s3: If True, get files from S3; if False, get from local filesystem
    
    Returns:
        List of file paths (local paths or S3 keys)
    """
    from utils.s3_service import s3_service
    
    if from_s3 and submission_id:
        # Get files from S3
        s3_prefix = f"lnh-submissions/{submission_id}/outputs/"
        s3_keys = s3_service.list_files(s3_prefix)
        
        print(f"📥 Looking for JSON files in S3 with prefix: {s3_prefix}")
        print(f"📦 Found {len(s3_keys)} files in S3")
        
        if not s3_keys:
            print(f"⚠️ No files found in S3 with prefix: {s3_prefix}")
            return []
        
        # Download files to temp directory for processing
        # Use a persistent temp directory that won't be cleaned up automatically
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f"lnh_summary_{submission_id}_")
        print(f"📁 Created temp directory: {temp_dir}")
        local_files = []
        
        for s3_key in s3_keys:
            if s3_key.endswith('.json'):
                filename = os.path.basename(s3_key)
                local_path = os.path.join(temp_dir, filename)
                print(f"📥 Downloading {s3_key} to {local_path}")
                if s3_service.download_file(s3_key, local_path):
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        local_files.append(local_path)
                        print(f"✅ Successfully downloaded {filename} ({os.path.getsize(local_path)} bytes)")
                    else:
                        print(f"⚠️ File {local_path} does not exist or is empty after download")
                else:
                    print(f"❌ Failed to download {s3_key}")
        
        print(f"📊 Total {len(local_files)} JSON files ready for processing")
        return local_files
    else:
        # Get files from local filesystem
        import glob
        search_dir = OUTPUT_DIR
        if submission_id:
            search_dir = os.path.join(OUTPUT_DIR, submission_id)
        
        if os.path.exists(search_dir):
            json_files = glob.glob(os.path.join(search_dir, "*.json"))
            return sorted(json_files, key=os.path.getmtime, reverse=True)
        else:
            return []

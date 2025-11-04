import boto3
import os
from typing import Optional
from config.settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# L&H Submission S3 Bucket
LN_H_SUBMISSION_S3_BUCKET = os.getenv("LN_H_SUBMISSION_S3_BUCKET", "lnh-submissions-production")

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        self.bucket = LN_H_SUBMISSION_S3_BUCKET
    
    def upload_file(self, file_path: str, s3_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file to S3
        
        Args:
            file_path: Local file path to upload
            s3_key: S3 key (path) where file will be stored
            content_type: Optional content type (e.g., 'application/json')
        
        Returns:
            S3 URL of the uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        self.s3_client.upload_file(
            file_path,
            self.bucket,
            s3_key,
            ExtraArgs=extra_args
        )
        
        # Return public URL
        url = f"https://{self.bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return url
    
    def upload_fileobj(self, file_obj, s3_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file-like object to S3
        
        Args:
            file_obj: File-like object (e.g., BytesIO, file handle)
            s3_key: S3 key (path) where file will be stored
            content_type: Optional content type (e.g., 'application/json')
        
        Returns:
            S3 URL of the uploaded file
        """
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        self.s3_client.upload_fileobj(
            file_obj,
            self.bucket,
            s3_key,
            ExtraArgs=extra_args
        )
        
        # Return public URL
        url = f"https://{self.bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return url
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 key (path) of the file to download
            local_path: Local file path where file will be saved
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return False
    
    def get_file_content(self, s3_key: str) -> Optional[bytes]:
        """
        Get file content from S3 as bytes
        
        Args:
            s3_key: S3 key (path) of the file
        
        Returns:
            File content as bytes, or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            return response['Body'].read()
        except Exception as e:
            print(f"Error getting file content from S3: {e}")
            return None
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key: S3 key (path) of the file
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception:
            return False
    
    def list_files(self, prefix: str) -> list:
        """
        List all files with a given prefix
        
        Args:
            prefix: S3 key prefix (e.g., 'lnh-submissions/{submission_id}/outputs/')
        
        Returns:
            List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"Error listing files from S3: {e}")
            return []

# Global S3 service instance
s3_service = S3Service()


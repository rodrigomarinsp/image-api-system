"""
storage.py - Cloud storage integration

This module handles integration with cloud storage for
storing and retrieving image files. Uses MinIO in development
and Google Cloud Storage in production.
"""
import os
import uuid
import logging
from typing import BinaryIO, Tuple, Optional
from datetime import datetime, timedelta

from fastapi import UploadFile, HTTPException, status
from PIL import Image as PILImage
import io

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Storage configuration
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "image-management-api-local")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# MinIO configuration for local development
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
USE_MINIO = os.getenv("USE_MINIO", "true").lower() == "true"

# Flag to determine if we're in local development or production
is_local_dev = USE_MINIO

# Initialize storage client
storage_client = None
boto3_client = None

# We'll use boto3 for MinIO in local development
if is_local_dev:
    try:
        import boto3
        from botocore.client import Config
        
        logger.info(f"Initializing MinIO storage client with endpoint: {MINIO_ENDPOINT}")
        boto3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # Check if bucket exists
        try:
            boto3_client.head_bucket(Bucket=BUCKET_NAME)
            logger.info(f"Using existing MinIO bucket: {BUCKET_NAME}")
        except Exception as e:
            # Create bucket if it doesn't exist
            try:
                boto3_client.create_bucket(Bucket=BUCKET_NAME)
                boto3_client.put_bucket_policy(
                    Bucket=BUCKET_NAME,
                    Policy=f'{{"Version":"2012-10-17","Statement":[{{"Effect":"Allow","Principal":{{"AWS":["*"]}},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::{BUCKET_NAME}/*"]}}]}}'
                )
                logger.info(f"Created new MinIO bucket: {BUCKET_NAME}")
            except Exception as bucket_e:
                logger.error(f"Error creating MinIO bucket: {str(bucket_e)}")
    except ImportError:
        logger.error("boto3 not installed - needed for local development with MinIO")
else:
    # Production mode - use Google Cloud Storage
    try:
        from google.cloud import storage
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            if not bucket.exists():
                bucket = storage_client.create_bucket(BUCKET_NAME)
                logger.info(f"Created new GCS bucket: {BUCKET_NAME}")
            else:
                logger.info(f"Using existing GCS bucket: {BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Error initializing GCS client: {str(e)}", exc_info=True)
            storage_client = None
    except ImportError:
        logger.error("google-cloud-storage not installed - needed for production mode")

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename"""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""

async def validate_image(file: UploadFile) -> Tuple[bool, str]:
    """
    Validate an uploaded image file
    Returns: (is_valid, error_message)
    """
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file position
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE/(1024*1024)}MB"
    
    # Check file extension
    extension = get_file_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"File extension not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Validate image data using PIL
    try:
        image_data = await file.read()
        img = PILImage.open(io.BytesIO(image_data))
        img.verify()  # Verify image integrity
        # Reset file position after reading
        await file.seek(0)
        return True, ""
    except Exception as e:
        await file.seek(0)
        return False, f"Invalid image data: {str(e)}"

async def upload_file_to_gcs(file: UploadFile, team_id: uuid.UUID) -> dict:
    """
    Upload a file to cloud storage
    Returns: metadata dictionary with storage information
    """
    global storage_client, boto3_client
    
    if not storage_client and not boto3_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud storage is not available"
        )
    
    # Validate image before upload
    is_valid, error_msg = await validate_image(file)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    # Generate unique filename
    extension = get_file_extension(file.filename)
    unique_id = str(uuid.uuid4())
    storage_filename = f"{team_id}/{unique_id}.{extension}"
    
    try:
        content = await file.read()
        
        if is_local_dev and boto3_client:
            # MinIO upload (local development)
            boto3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=storage_filename,
                Body=content,
                ContentType=file.content_type
            )
            # Generate storage path for MinIO
            storage_path = f"minio://{BUCKET_NAME}/{storage_filename}"
            logger.info(f"Uploaded file to MinIO: {storage_filename}")
        else:
            # Google Cloud Storage upload (production)
            blob = storage_client.bucket(BUCKET_NAME).blob(storage_filename)
            blob.upload_from_string(content, content_type=file.content_type)
            storage_path = f"gs://{BUCKET_NAME}/{storage_filename}"
            logger.info(f"Uploaded file to GCS: {storage_filename}")
        
        # Generate metadata
        metadata = {
            "filename": storage_filename,
            "storage_path": storage_path,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "original_filename": file.filename
        }
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )

def generate_signed_url(storage_path: str, expires_in_minutes: int = 30) -> str:
    """Generate a signed URL for temporary access to a file"""
    global storage_client, boto3_client
    
    if not storage_client and not boto3_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud storage is not available"
        )
    
    try:
        # Handle different storage path formats
        if storage_path.startswith("gs://"):
            # Google Cloud Storage path
            if not storage_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Google Cloud Storage client not available"
                )
            
            # Extract blob path from gs:// URL
            parts = storage_path.replace("gs://", "").split("/", 1)
            if len(parts) > 1:
                bucket_name, blob_name = parts
            else:
                bucket_name, blob_name = BUCKET_NAME, parts[0]
            
            blob = storage_client.bucket(bucket_name).blob(blob_name)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expires_in_minutes),
                method="GET"
            )
            
        elif storage_path.startswith("minio://"):
            # MinIO path
            if not boto3_client:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MinIO client not available"
                )
                
            # Extract object path from minio:// URL
            parts = storage_path.replace("minio://", "").split("/", 1)
            if len(parts) > 1:
                bucket_name, object_name = parts
            else:
                bucket_name, object_name = BUCKET_NAME, parts[0]
            
            # Generate pre-signed URL
            url = boto3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expires_in_minutes * 60
            )
            
        else:
            # Assume it's a direct object key
            if is_local_dev and boto3_client:
                url = boto3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': BUCKET_NAME,
                        'Key': storage_path
                    },
                    ExpiresIn=expires_in_minutes * 60
                )
            else:
                blob = storage_client.bucket(BUCKET_NAME).blob(storage_path)
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(minutes=expires_in_minutes),
                    method="GET"
                )
                
        return url
        
    except Exception as e:
        logger.error(f"Error generating signed URL: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating access URL: {str(e)}"
        )

async def delete_file_from_gcs(storage_path: str) -> bool:
    """Delete a file from cloud storage"""
    global storage_client, boto3_client
    
    if not storage_client and not boto3_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud storage is not available"
        )
    
    try:
        # Handle different storage path formats
        if storage_path.startswith("gs://"):
            # Google Cloud Storage path
            if not storage_client:
                raise ValueError("Google Cloud Storage client not available")
                
            # Extract blob path from gs:// URL
            parts = storage_path.replace("gs://", "").split("/", 1)
            if len(parts) > 1:
                bucket_name, blob_name = parts
            else:
                bucket_name, blob_name = BUCKET_NAME, parts[0]
                
            blob = storage_client.bucket(bucket_name).blob(blob_name)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted file from GCS: {blob_name}")
                return True
            else:
                logger.warning(f"File not found in GCS for deletion: {blob_name}")
                return False
                
        elif storage_path.startswith("minio://"):
            # MinIO path
            if not boto3_client:
                raise ValueError("MinIO client not available")
                
            # Extract object path from minio:// URL
            parts = storage_path.replace("minio://", "").split("/", 1)
            if len(parts) > 1:
                bucket_name, object_name = parts
            else:
                bucket_name, object_name = BUCKET_NAME, parts[0]
                
            # Delete object
            boto3_client.delete_object(
                Bucket=bucket_name,
                Key=object_name
            )
            logger.info(f"Deleted file from MinIO: {object_name}")
            return True
            
        else:
            # Assume it's a direct object key
            if is_local_dev and boto3_client:
                boto3_client.delete_object(
                    Bucket=BUCKET_NAME,
                    Key=storage_path
                )
                logger.info(f"Deleted file from MinIO: {storage_path}")
                return True
            else:
                blob = storage_client.bucket(BUCKET_NAME).blob(storage_path)
                if blob.exists():
                    blob.delete()
                    logger.info(f"Deleted file from GCS: {storage_path}")
                    return True
                else:
                    logger.warning(f"File not found in GCS for deletion: {storage_path}")
                    return False
    
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )

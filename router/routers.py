"""
routers.py - API endpoints and routing

This module contains all FastAPI routers and endpoints for the Image Management API.
"""
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Request, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, Field, EmailStr

from db.database import get_db, Team, User, APIKey, Image, ImageEmbedding, AccessLog
from auth.auth import get_current_user, generate_api_key, verify_team_access
from services.storage import upload_file_to_gcs, generate_signed_url, delete_file_from_gcs
from services.image_understanding import generate_image_embedding, semantic_search

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for API
class TeamBase(BaseModel):
    name: str
    description: Optional[str] = None

class TeamCreate(TeamBase):
    pass

class TeamResponse(TeamBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserBase(BaseModel):
    username: str
    email: EmailStr
    team_id: uuid.UUID
    is_admin: bool = False

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class APIKeyBase(BaseModel):
    name: str
    expires_at: Optional[datetime] = None

class APIKeyCreate(APIKeyBase):
    pass

class APIKeyResponse(APIKeyBase):
    id: uuid.UUID
    key: str  # Only returned on creation
    user_id: uuid.UUID
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ImageMetadata(BaseModel):
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    alt_text: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None

class ImageResponse(BaseModel):
    id: uuid.UUID
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    team_id: uuid.UUID
    uploaded_by_user_id: uuid.UUID
    metadata: Optional[ImageMetadata] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    access_url: Optional[str] = None
    has_embedding: bool = False
    
    class Config:
        from_attributes = True

class SearchQuery(BaseModel):
    prompt: str
    limit: int = 10

class SearchResult(BaseModel):
    image_id: uuid.UUID
    similarity_score: float
    access_url: str
    metadata: Optional[ImageMetadata] = None

# Create routers
team_router = APIRouter()
user_router = APIRouter()
api_key_router = APIRouter()
image_router = APIRouter()
search_router = APIRouter()

# Team endpoints
@team_router.post("/teams", response_model=TeamResponse)
async def create_team(
    team: TeamCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new team"""
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create teams"
        )
    
    # Create team
    new_team = Team(**team.dict())
    db.add(new_team)
    await db.commit()
    await db.refresh(new_team)
    
    logger.info(f"Team created: {new_team.id}")
    
    # Convert SQLAlchemy model to dictionary before returning
    team_dict = {
        "id": new_team.id,
        "name": new_team.name,
        "description": new_team.description,
        "created_at": new_team.created_at,
        "updated_at": new_team.updated_at
    }
    
    return team_dict

@team_router.get("/teams", response_model=List[TeamResponse])
async def list_teams(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all teams (admin) or user's team"""
    if current_user.is_admin:
        # Admin can see all teams
        result = await db.execute(select(Team))
        teams = result.scalars().all()
    else:
        # Regular users can only see their team
        result = await db.execute(
            select(Team).where(Team.id == current_user.team_id)
        )
        teams = result.scalars().all()
    
    # Convert SQLAlchemy objects to dictionaries
    team_dicts = []
    for team in teams:
        team_dict = {
            "id": team.id,
            "name": team.name,
            "description": team.description,
            "created_at": team.created_at,
            "updated_at": team.updated_at
        }
        team_dicts.append(team_dict)
    
    return team_dicts

@team_router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a team by ID"""
    # Check access permissions
    if not await verify_team_access(current_user, team_id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this team"
        )
    
    # Get team
    result = await db.execute(
        select(Team).where(Team.id == team_id)
    )
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found"
        )
    
    # Convert SQLAlchemy model to dictionary before returning
    team_dict = {
        "id": team.id,
        "name": team.name,
        "description": team.description,
        "created_at": team.created_at,
        "updated_at": team.updated_at
    }
    
    return team_dict

# User endpoints
@user_router.post("/users", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new user"""
    # Check if admin or same team
    if not current_user.is_admin and current_user.team_id != user.team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create users in your own team"
        )
    
    # Check if team exists
    result = await db.execute(
        select(Team).where(Team.id == user.team_id)
    )
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found"
        )
    
    # Create user
    new_user = User(**user.dict())
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"User created: {new_user.id}")
    
    # Convert SQLAlchemy model to dictionary before returning
    user_dict = {
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "team_id": new_user.team_id,
        "is_admin": new_user.is_admin,
        "created_at": new_user.created_at,
        "updated_at": new_user.updated_at
    }
    
    return user_dict

@user_router.get("/users", response_model=List[UserResponse])
async def list_users(
    team_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List users with optional team filtering"""
    # Determine which teams to filter by
    if team_id:
        # If specific team requested, check access
        if not await verify_team_access(current_user, team_id) and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this team's users"
            )
        filter_team_id = team_id
    elif current_user.is_admin:
        # Admin with no filter gets all users
        filter_team_id = None
    else:
        # Non-admin gets only their team
        filter_team_id = current_user.team_id
    
    # Execute query
    if filter_team_id:
        result = await db.execute(
            select(User).where(User.team_id == filter_team_id)
        )
    else:
        result = await db.execute(select(User))
    
    users = result.scalars().all()
    
    # Convert SQLAlchemy objects to dictionaries
    user_dicts = []
    for user in users:
        user_dict = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "team_id": user.team_id,
            "is_admin": user.is_admin,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
        user_dicts.append(user_dict)
    
    return user_dicts

@user_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a user by ID"""
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check access permissions
    if not current_user.is_admin and current_user.team_id != user.team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this user"
        )
    
    # Convert SQLAlchemy model to dictionary before returning
    user_dict = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "team_id": user.team_id,
        "is_admin": user.is_admin,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }
    
    return user_dict

@user_router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get the current authenticated user's information"""
    # Convert SQLAlchemy model to dictionary before returning
    user_dict = {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "team_id": current_user.team_id,
        "is_admin": current_user.is_admin,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }
    
    return user_dict

# API Key endpoints
@api_key_router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new API key for the current user"""
    # Generate a new API key
    key = generate_api_key()
    
    # Create API key record
    new_api_key = APIKey(
        key=key,
        user_id=current_user.id,
        **api_key.dict()
    )
    
    db.add(new_api_key)
    await db.commit()
    await db.refresh(new_api_key)
    
    logger.info(f"API key created for user: {current_user.id}")
    
    # Convert SQLAlchemy model to dictionary before returning
    api_key_dict = {
        "id": new_api_key.id,
        "key": new_api_key.key,
        "name": new_api_key.name,
        "user_id": new_api_key.user_id,
        "is_active": new_api_key.is_active,
        "expires_at": new_api_key.expires_at,
        "created_at": new_api_key.created_at,
        "last_used_at": new_api_key.last_used_at
    }
    
    return api_key_dict

@api_key_router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List API keys for the current user"""
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == current_user.id)
    )
    api_keys = result.scalars().all()
    
    # Convert SQLAlchemy objects to dictionaries
    api_key_dicts = []
    for api_key in api_keys:
        # Filter out actual key values for security
        masked_key = "***" + api_key.key[-4:]
        
        api_key_dict = {
            "id": api_key.id,
            "key": masked_key,
            "name": api_key.name,
            "user_id": api_key.user_id,
            "is_active": api_key.is_active,
            "expires_at": api_key.expires_at,
            "created_at": api_key.created_at,
            "last_used_at": api_key.last_used_at
        }
        api_key_dicts.append(api_key_dict)
    
    return api_key_dicts

@api_key_router.delete("/api-keys/{api_key_id}")
async def revoke_api_key(
    api_key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke (deactivate) an API key"""
    # Get API key
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == api_key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or not owned by you"
        )
    
    # Revoke the key
    api_key.is_active = False
    await db.commit()
    
    logger.info(f"API key revoked: {api_key_id}")
    return {"message": "API key revoked successfully"}

# Image endpoints
@image_router.post("/images", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    alt_text: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a new image"""
    # Upload file to cloud storage
    storage_metadata = await upload_file_to_gcs(file, current_user.team_id)
    
    # Prepare metadata
    metadata = {
        "description": description,
        "alt_text": alt_text,
    }
    
    # Parse tags if provided
    if tags:
        metadata["tags"] = [tag.strip() for tag in tags.split(",")]
    
    # Create image record
    new_image = Image(
        filename=storage_metadata["filename"],
        storage_path=storage_metadata["storage_path"],
        content_type=storage_metadata["content_type"],
        size_bytes=storage_metadata["size_bytes"],
        original_filename=storage_metadata["original_filename"],
        team_id=current_user.team_id,
        uploaded_by_user_id=current_user.id,
        image_metadata=metadata
    )
    
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)
    
    logger.info(f"Image uploaded: {new_image.id}")
    
    # Generate temporary access URL
    access_url = generate_signed_url(new_image.storage_path)
    
    # Return response with properly mapped fields
    image_dict = {
        "id": new_image.id,
        "filename": new_image.filename,
        "original_filename": new_image.original_filename,
        "content_type": new_image.content_type,
        "size_bytes": new_image.size_bytes,
        "team_id": new_image.team_id,
        "uploaded_by_user_id": new_image.uploaded_by_user_id,
        "metadata": new_image.image_metadata,  # Map image_metadata to metadata
        "created_at": new_image.created_at,
        "updated_at": new_image.updated_at,
        "access_url": access_url,
        "has_embedding": False
    }
    
    return image_dict

@image_router.get("/images", response_model=List[ImageResponse])
async def list_images(
    team_id: Optional[uuid.UUID] = None,
    tag: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List images with optional filtering
    - Filter by team (with permission check)
    - Filter by tag
    - Pagination with limit/offset
    """
    # Determine which teams to filter by
    if team_id:
        # If specific team requested, check access
        if not await verify_team_access(current_user, team_id) and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this team's images"
            )
        filter_team_id = team_id
    else:
        # Default to user's team
        filter_team_id = current_user.team_id
    
    # Base query
    query = select(Image).where(Image.team_id == filter_team_id)
    
    # Add tag filter if provided
    if tag:
        # Filter on the JSONB metadata field, checking if the tag is in the tags array
        # Note: The exact SQL syntax might vary by database, this is PostgreSQL syntax
        query = query.filter(Image.image_metadata["tags"].contains([tag]))
    
    # Add pagination
    query = query.offset(offset).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    images = result.scalars().all()
    
    # Check for embeddings and add access URLs
    enhanced_images = []
    for image in images:
        # Check if image has embeddings
        embedding_result = await db.execute(
            select(ImageEmbedding).where(ImageEmbedding.image_id == image.id)
        )
        has_embedding = embedding_result.scalar_one_or_none() is not None
        
        # Generate access URL
        access_url = generate_signed_url(image.storage_path)
        
        # Add to enhanced results with proper field mapping
        enhanced_images.append({
            "id": image.id,
            "filename": image.filename,
            "original_filename": image.original_filename,
            "content_type": image.content_type,
            "size_bytes": image.size_bytes,
            "team_id": image.team_id,
            "uploaded_by_user_id": image.uploaded_by_user_id,
            "metadata": image.image_metadata,  # Map image_metadata to metadata
            "created_at": image.created_at,
            "updated_at": image.updated_at,
            "access_url": access_url,
            "has_embedding": has_embedding
        })
    
    return enhanced_images

@image_router.get("/images/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get image details by ID"""
    # Get image
    result = await db.execute(
        select(Image).where(Image.id == image_id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Check team access
    if not await verify_team_access(current_user, image.team_id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this image"
        )
    
    # Check if image has embeddings
    embedding_result = await db.execute(
        select(ImageEmbedding).where(ImageEmbedding.image_id == image.id)
    )
    has_embedding = embedding_result.scalar_one_or_none() is not None
    
    # Generate access URL
    access_url = generate_signed_url(image.storage_path)
    
    # We need to manually create a dictionary with the correct property names
    image_dict = {
        "id": image.id,
        "filename": image.filename,
        "original_filename": image.original_filename,
        "content_type": image.content_type,
        "size_bytes": image.size_bytes,
        "team_id": image.team_id,
        "uploaded_by_user_id": image.uploaded_by_user_id,
        "metadata": image.image_metadata,  # Map image_metadata to metadata
        "created_at": image.created_at,
        "updated_at": image.updated_at,
        "access_url": access_url,
        "has_embedding": has_embedding
    }
    
    return image_dict

@image_router.delete("/images/{image_id}")
async def delete_image(
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete an image"""
    # Get image
    result = await db.execute(
        select(Image).where(Image.id == image_id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Check team access
    if not await verify_team_access(current_user, image.team_id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to delete this image"
        )
    
    # Delete from cloud storage
    await delete_file_from_gcs(image.storage_path)
    
    # Delete any embeddings
    await db.execute(
        "DELETE FROM image_embeddings WHERE image_id = :image_id",
        {"image_id": image_id}
    )
    
    # Delete from database
    await db.execute(
        "DELETE FROM images WHERE id = :image_id",
        {"image_id": image_id}
    )
    await db.commit()
    
    logger.info(f"Image deleted: {image_id}")
    return {"message": "Image deleted successfully"}

@image_router.post("/images/{image_id}/embedding")
async def create_image_embedding(
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate and store embeddings for an image"""
    # Get image
    result = await db.execute(
        select(Image).where(Image.id == image_id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Check team access
    if not await verify_team_access(current_user, image.team_id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this image"
        )
    
    # Check if embedding already exists
    embedding_result = await db.execute(
        select(ImageEmbedding).where(ImageEmbedding.image_id == image_id)
    )
    existing_embedding = embedding_result.scalar_one_or_none()
    
    if existing_embedding:
        return {"message": "Embedding already exists for this image"}
    
    # Generate embedding
    try:
        # Generate signed URL for the image embedding service
        access_url = generate_signed_url(image.storage_path)
        
        # Generate embedding
        embedding_data = await generate_image_embedding(image_id, access_url)
        
        # Store embedding
        new_embedding = ImageEmbedding(
            image_id=image_id,
            embedding_vector=embedding_data["embedding"],
            embedding_model=embedding_data["model"],
            vector_db_id=embedding_data.get("vector_db_id")
        )
        
        db.add(new_embedding)
        await db.commit()
        
        logger.info(f"Image embedding created: {image_id}")
        return {"message": "Image embedding created successfully"}
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embedding: {str(e)}"
        )

# Semantic search endpoints
@search_router.post("/search", response_model=List[SearchResult])
async def search_images(
    query: SearchQuery,
    team_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search for images using semantic similarity to a text prompt"""
    # Determine which team to search in
    if team_id:
        # If specific team requested, check access
        if not await verify_team_access(current_user, team_id) and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this team's images"
            )
        filter_team_id = team_id
    else:
        # Default to user's team
        filter_team_id = current_user.team_id
    
    try:
        # Perform semantic search
        search_results = await semantic_search(query.prompt, filter_team_id, query.limit)
        
        # Enhance results with access URLs
        enhanced_results = []
        for result in search_results:
            # Get image information
            image_result = await db.execute(
                select(Image).where(Image.id == result["image_id"])
            )
            image = image_result.scalar_one_or_none()
            
            if image:
                # Generate access URL
                access_url = generate_signed_url(image.storage_path)
                
                # Add to enhanced results
                enhanced_results.append({
                    "image_id": result["image_id"],
                    "similarity_score": result["similarity_score"],
                    "access_url": access_url,
                    "metadata": image.image_metadata  # Fix: use image_metadata instead of metadata
                })
        
        return enhanced_results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )

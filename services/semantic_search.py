"""
semantic_search.py - Enhanced Semantic Search Functionality

This module provides advanced semantic search capabilities for images using
ChromaDB and various embedding models. It supports text-to-image search,
image-to-image search, and multimodal search with filtering options.
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import httpx
import asyncio
import json
from PIL import Image as PILImage
import io

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from other modules
from image_understanding import embedding_model, EMBEDDING_DIMENSION, ENABLE_REAL_EMBEDDINGS
from db.vector_db import search_by_embedding, get_collection_stats

# Configuration
SEARCH_CACHE_EXPIRY = int(os.getenv("SEARCH_CACHE_EXPIRY", "3600"))  # Cache expiry in seconds
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "100"))  # Default max results
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))  # Min similarity score (0-1)

# Cache for search results
search_cache = {}

class SearchQuery:
    """Class representing a semantic search query with various parameters"""
    
    def __init__(
        self,
        prompt: Optional[str] = None,
        reference_image_id: Optional[uuid.UUID] = None,
        reference_image_url: Optional[str] = None,
        team_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        filter_similar_images: bool = False,
        min_similarity: float = SIMILARITY_THRESHOLD
    ):
        self.prompt = prompt
        self.reference_image_id = reference_image_id
        self.reference_image_url = reference_image_url
        self.team_id = team_id
        self.tags = tags or []
        self.date_from = date_from
        self.date_to = date_to
        self.limit = min(limit, SEARCH_RESULT_LIMIT)
        self.offset = offset
        self.filter_similar_images = filter_similar_images
        self.min_similarity = min_similarity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search query to dictionary"""
        return {
            "prompt": self.prompt,
            "reference_image_id": str(self.reference_image_id) if self.reference_image_id else None,
            "reference_image_url": self.reference_image_url,
            "team_id": str(self.team_id) if self.team_id else None,
            "tags": self.tags,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "limit": self.limit,
            "offset": self.offset,
            "filter_similar_images": self.filter_similar_images,
            "min_similarity": self.min_similarity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchQuery':
        """Create search query from dictionary"""
        return cls(
            prompt=data.get("prompt"),
            reference_image_id=uuid.UUID(data["reference_image_id"]) if data.get("reference_image_id") else None,
            reference_image_url=data.get("reference_image_url"),
            team_id=uuid.UUID(data["team_id"]) if data.get("team_id") else None,
            tags=data.get("tags", []),
            date_from=data.get("date_from"),
            date_to=data.get("date_to"),
            limit=data.get("limit", 10),
            offset=data.get("offset", 0),
            filter_similar_images=data.get("filter_similar_images", False),
            min_similarity=data.get("min_similarity", SIMILARITY_THRESHOLD)
        )
    
    def get_cache_key(self) -> str:
        """Generate a cache key for this search query"""
        key_parts = [
            self.prompt or "",
            str(self.reference_image_id or ""),
            str(self.team_id or ""),
            ",".join(sorted(self.tags)) if self.tags else "",
            self.date_from or "",
            self.date_to or "",
            str(self.min_similarity)
        ]
        return ":".join(key_parts)

class SearchResult:
    """Class representing a semantic search result"""
    
    def __init__(
        self,
        image_id: uuid.UUID,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None,
        access_url: Optional[str] = None
    ):
        self.image_id = image_id
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        self.access_url = access_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary"""
        return {
            "image_id": str(self.image_id),
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "access_url": self.access_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create search result from dictionary"""
        return cls(
            image_id=uuid.UUID(data["image_id"]),
            similarity_score=data["similarity_score"],
            metadata=data.get("metadata", {}),
            access_url=data.get("access_url")
        )

async def get_embedding_from_text(text: str) -> List[float]:
    """Generate embedding vector from text"""
    if not ENABLE_REAL_EMBEDDINGS or not embedding_model:
        # Generate mock embedding for text
        import hashlib
        # Use hash of text to generate deterministic mock embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to list of floats in range [-1, 1]
        mock_embedding = []
        for i in range(0, len(text_hash), 2):
            if i+2 <= len(text_hash):
                hex_val = text_hash[i:i+2]
                float_val = (int(hex_val, 16) / 255.0) * 2 - 1
                mock_embedding.append(float_val)
        
        # Pad or trim to EMBEDDING_DIMENSION
        if len(mock_embedding) < EMBEDDING_DIMENSION:
            mock_embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(mock_embedding)))
        else:
            mock_embedding = mock_embedding[:EMBEDDING_DIMENSION]
            
        return mock_embedding
    
    # Generate real embedding using model
    try:
        embedding = embedding_model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}", exc_info=True)
        raise

async def get_embedding_from_image_url(image_url: str) -> List[float]:
    """Generate embedding vector from image URL"""
    if not ENABLE_REAL_EMBEDDINGS or not embedding_model:
        # Generate mock embedding for image
        import hashlib
        # Use hash of URL to generate deterministic mock embedding
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        # Convert hash to list of floats in range [-1, 1]
        mock_embedding = []
        for i in range(0, len(url_hash), 2):
            if i+2 <= len(url_hash):
                hex_val = url_hash[i:i+2]
                float_val = (int(hex_val, 16) / 255.0) * 2 - 1
                mock_embedding.append(float_val)
        
        # Pad or trim to EMBEDDING_DIMENSION
        if len(mock_embedding) < EMBEDDING_DIMENSION:
            mock_embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(mock_embedding)))
        else:
            mock_embedding = mock_embedding[:EMBEDDING_DIMENSION]
            
        return mock_embedding
    
    # Download and process image
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            image_data = response.content
            
        # Generate embedding
        embedding = embedding_model.encode(image_data, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating image embedding from URL: {str(e)}", exc_info=True)
        raise

async def get_embedding_for_search_query(query: SearchQuery) -> List[float]:
    """Get embedding vector for a search query"""
    # Text-based search
    if query.prompt:
        return await get_embedding_from_text(query.prompt)
    
    # Image-based search by ID
    elif query.reference_image_id:
        # Get image URL from database (implement as needed)
        # For now, we'll just use a mock method
        image_url = await get_image_url_from_id(query.reference_image_id)
        return await get_embedding_from_image_url(image_url)
    
    # Image-based search by URL
    elif query.reference_image_url:
        return await get_embedding_from_image_url(query.reference_image_url)
    
    else:
        raise ValueError("Search query must contain either prompt, reference_image_id, or reference_image_url")

async def get_image_url_from_id(image_id: uuid.UUID) -> str:
    """Get image URL from image ID (mock implementation)"""
    # In a real implementation, this would query the database
    # For now, return a mock URL
    return f"https://example.com/images/{image_id}.jpg"

async def enhance_search_results(
    results: List[Dict[str, Any]], 
    include_access_url: bool = True
) -> List[SearchResult]:
    """Enhance search results with additional information"""
    enhanced_results = []
    
    for result in results:
        image_id = result["image_id"]
        similarity_score = result["similarity_score"]
        metadata = result.get("metadata", {})
        
        # Add access URL if requested
        access_url = None
        if include_access_url:
            try:
                # In a real implementation, generate signed URL
                access_url = f"https://example.com/images/{image_id}.jpg"
            except Exception as e:
                logger.error(f"Error generating access URL: {str(e)}")
        
        # Create SearchResult object
        search_result = SearchResult(
            image_id=image_id,
            similarity_score=similarity_score,
            metadata=metadata,
            access_url=access_url
        )
        
        enhanced_results.append(search_result)
    
    return enhanced_results

async def perform_semantic_search(query: SearchQuery) -> List[SearchResult]:
    """Perform semantic search based on the given query"""
    cache_key = query.get_cache_key()
    
    # Check cache first
    if cache_key in search_cache:
        cache_entry = search_cache[cache_key]
        cache_time = cache_entry["timestamp"]
        cache_age = (datetime.now() - cache_time).total_seconds()
        
        # Return cached results if still valid
        if cache_age < SEARCH_CACHE_EXPIRY:
            logger.info(f"Returning cached search results for key: {cache_key}")
            results = cache_entry["results"]
            # Apply pagination
            paginated_results = results[query.offset:query.offset + query.limit]
            return [SearchResult.from_dict(r) for r in paginated_results]
    
    # Generate embedding for search query
    try:
        embedding = await get_embedding_for_search_query(query)
        
        # Perform search in vector database
        raw_results = await search_by_embedding(
            query_embedding=embedding,
            team_id=query.team_id,
            limit=SEARCH_RESULT_LIMIT  # Get more results for filtering and pagination
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in raw_results 
            if r["similarity_score"] >= query.min_similarity
        ]
        
        # Enhance results with additional information
        enhanced_results = await enhance_search_results(filtered_results)
        
        # Store in cache
        search_cache[cache_key] = {
            "timestamp": datetime.now(),
            "results": [r.to_dict() for r in enhanced_results]
        }
        
        # Apply pagination
        paginated_results = enhanced_results[query.offset:query.offset + query.limit]
        
        logger.info(f"Semantic search completed with {len(enhanced_results)} results, returning {len(paginated_results)}")
        return paginated_results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
        raise

async def find_similar_images(
    image_id: uuid.UUID,
    team_id: Optional[uuid.UUID] = None,
    limit: int = 10,
    min_similarity: float = SIMILARITY_THRESHOLD
) -> List[SearchResult]:
    """Find images similar to the specified image"""
    try:
        # Create search query
        query = SearchQuery(
            reference_image_id=image_id,
            team_id=team_id,
            limit=limit,
            min_similarity=min_similarity
        )
        
        # Perform search
        return await perform_semantic_search(query)
    
    except Exception as e:
        logger.error(f"Error finding similar images: {str(e)}", exc_info=True)
        raise

async def semantic_search_with_filters(
    query: SearchQuery
) -> Tuple[List[SearchResult], Dict[str, Any]]:
    """
    Perform semantic search with additional filtering options
    Returns: Tuple of (results, stats)
    """
    try:
        # Perform basic semantic search
        results = await perform_semantic_search(query)
        
        # Calculate statistics
        stats = {
            "total_results": len(results),
            "avg_similarity": sum(r.similarity_score for r in results) / len(results) if results else 0,
            "max_similarity": max(r.similarity_score for r in results) if results else 0,
            "min_similarity": min(r.similarity_score for r in results) if results else 0,
            "query_type": "text" if query.prompt else "image"
        }
        
        return results, stats
    
    except Exception as e:
        logger.error(f"Error in semantic search with filters: {str(e)}", exc_info=True)
        raise

def clear_search_cache():
    """Clear the search cache"""
    global search_cache
    search_cache = {}
    logger.info("Search cache cleared")

async def get_search_stats() -> Dict[str, Any]:
    """Get statistics about semantic search functionality"""
    try:
        # Get vector database stats
        vector_stats = get_collection_stats()
        
        # Get embedding model stats
        model_stats = {
            "model": EMBEDDING_MODEL,
            "enabled": ENABLE_REAL_EMBEDDINGS,
            "dimension": EMBEDDING_DIMENSION
        }
        
        # Get cache stats
        cache_stats = {
            "entries": len(search_cache),
            "size_bytes": sum(len(json.dumps(entry)) for entry in search_cache.values()) if search_cache else 0,
            "expiry_seconds": SEARCH_CACHE_EXPIRY
        }
        
        return {
            "vector_database": vector_stats,
            "embedding_model": model_stats,
            "search_cache": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting search stats: {str(e)}", exc_info=True)
        return {"error": str(e)}

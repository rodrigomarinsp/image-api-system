"""
image_understanding.py - Image understanding and semantic search

This module handles image understanding using computer vision models
and implements semantic search functionality with ChromaDB vector database.
"""
import os
import uuid
import logging
import numpy as np
from typing import List, Dict, Any
import httpx
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Disable verbose tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "clip")  # CLIP or other models
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "512"))  # Model-specific
ENABLE_REAL_EMBEDDINGS = os.getenv("ENABLE_REAL_EMBEDDINGS", "false").lower() == "true"

# Global variables for models
embedding_model = None

# Initialize embedding model
def init_embedding_model():
    global embedding_model
    
    if not ENABLE_REAL_EMBEDDINGS:
        logger.warning("Real embeddings disabled. Will use mock embeddings.")
        return False
    
    try:
        if EMBEDDING_MODEL.lower() == "clip":
            from sentence_transformers import SentenceTransformer
            
            # Load CLIP model
            embedding_model = SentenceTransformer("clip-ViT-B-32")
            logger.info("CLIP model loaded successfully")
            return True
        else:
            logger.error(f"Unsupported embedding model: {EMBEDDING_MODEL}")
            return False
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        return False

# Try to initialize on module import
try:
    init_embedding_model()
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}", exc_info=True)

async def generate_image_embedding(image_id: uuid.UUID, image_url: str) -> Dict[str, Any]:
    """
    Generate embedding for an image using the configured model
    Returns: Dictionary with embedding vector and model information
    """
    if not ENABLE_REAL_EMBEDDINGS or not embedding_model:
        # Use mock embeddings if real embeddings are disabled or model not available
        logger.warning(f"Using mock embedding for image {image_id}")
        mock_embedding = [float(i % 10) / 10 for i in range(EMBEDDING_DIMENSION)]
        
        # Import the store_embedding function
        from db.vector_db import store_embedding
        
        # Store embedding in ChromaDB
        metadata = {
            "image_id": str(image_id),
            "created_at": datetime.now().isoformat(),
            "original_filename": "mock_image.jpg",
            "content_type": "image/jpeg"
        }
        
        await store_embedding(image_id, mock_embedding, metadata)
        
        return {
            "embedding": mock_embedding,
            "model": f"mock-{EMBEDDING_MODEL}",
            "vector_db_id": str(image_id)
        }
    
    try:
        # Download image
        async with httpx.AsyncClient() as client:
            logger.info(f"Downloading image from {image_url}")
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            image_data = response.content
        
        # Generate embedding
        logger.info(f"Generating embedding for image {image_id}")
        embedding = embedding_model.encode(image_data, show_progress_bar=False)
        embedding_list = embedding.tolist()
        
        # Import the store_embedding function
        from db.vector_db import store_embedding
        
        # Extract image metadata
        metadata = {
            "image_id": str(image_id),
            "created_at": datetime.now().isoformat()
        }
        
        # Add image metadata if available from URL context
        try:
            async with httpx.AsyncClient() as client:
                db_result = await client.get(f"/api/v1/images/{image_id}", headers={"Authorization": "direct_api_key_123456"})
                if db_result.status_code == 200:
                    image_data = db_result.json()
                    metadata["original_filename"] = image_data.get("original_filename", "")
                    metadata["content_type"] = image_data.get("content_type", "")
                    metadata["team_id"] = image_data.get("team_id", "")
        except Exception as metadata_e:
            logger.error(f"Error fetching image metadata: {str(metadata_e)}", exc_info=True)
        
        # Store embedding in ChromaDB
        success = await store_embedding(image_id, embedding_list, metadata)
        if success:
            logger.info(f"Successfully stored embedding for image {image_id} in ChromaDB")
        else:
            logger.warning(f"Failed to store embedding for image {image_id} in ChromaDB")
        
        return {
            "embedding": embedding_list,
            "model": EMBEDDING_MODEL,
            "vector_db_id": str(image_id)
        }
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise

async def semantic_search(
    prompt: str, 
    team_id: uuid.UUID, 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using a text prompt
    Returns: List of image IDs and similarity scores
    """
    if not ENABLE_REAL_EMBEDDINGS or not embedding_model:
        # Use mock search results if vector search is not fully available
        logger.warning(f"Using mock search results for prompt: {prompt}")
        
        # Generate mock search results
        import random
        mock_results = []
        for i in range(min(limit, 5)):
            mock_results.append({
                "image_id": uuid.uuid4(),
                "similarity_score": round(0.9 - (i * 0.15), 2)  # Decreasing scores
            })
        
        return mock_results
    
    try:
        # Generate text embedding from prompt
        logger.info(f"Generating embedding for search prompt: {prompt}")
        text_embedding = embedding_model.encode([prompt], show_progress_bar=False)[0]
        
        # Import search function from vector_db
        from db.vector_db import search_by_embedding
        
        # Perform search
        logger.info(f"Searching for images similar to prompt: {prompt}")
        results = await search_by_embedding(
            query_embedding=text_embedding.tolist(),
            team_id=team_id,
            limit=limit
        )
        
        logger.info(f"Found {len(results)} results for prompt: {prompt}")
        return results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
        raise

def get_embedding_stats():
    """Get statistics about the embedding model"""
    return {
        "model_type": EMBEDDING_MODEL,
        "model_loaded": embedding_model is not None,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "real_embeddings_enabled": ENABLE_REAL_EMBEDDINGS
    }

async def process_image_batch(image_ids_and_urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of images to generate embeddings"""
    results = []
    for item in image_ids_and_urls:
        try:
            embedding = await generate_image_embedding(
                uuid.UUID(item["image_id"]), 
                item["image_url"]
            )
            results.append({
                "image_id": item["image_id"],
                "success": True,
                "vector_db_id": embedding.get("vector_db_id")
            })
        except Exception as e:
            results.append({
                "image_id": item["image_id"],
                "success": False,
                "error": str(e)
            })
    return results

"""
vector_db.py - ChromaDB Vector Database Integration

This module handles the vector database functionality using ChromaDB.
It provides functions for storing and retrieving image embeddings.
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "image_embeddings")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "512"))

# Global ChromaDB client and collection
chroma_client = None
collection = None

# Flag to track if vector DB is available
vector_db_available = False

def init_vector_db():
    """Initialize the ChromaDB vector database"""
    global chroma_client, collection, vector_db_available
    
    try:
        # Try to import ChromaDB
        try:
            import chromadb
            logger.info("Successfully imported ChromaDB")
        except ImportError as e:
            logger.error(f"Failed to import ChromaDB: {str(e)}")
            logger.warning("Vector database functionality will be disabled")
            vector_db_available = False
            return False
        
        # Create persistence directory if it doesn't exist
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            logger.warning("Attempting to use in-memory client as fallback")
            try:
                chroma_client = chromadb.Client()
            except Exception as e2:
                logger.error(f"Failed to initialize in-memory ChromaDB client: {str(e2)}")
                vector_db_available = False
                return False
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            logger.info(f"Using existing ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        except Exception as e:
            try:
                # Create a new collection
                collection = chroma_client.create_collection(
                    name=CHROMA_COLLECTION_NAME,
                    metadata={"description": "Image embeddings for semantic search"}
                )
                logger.info(f"Created new ChromaDB collection: {CHROMA_COLLECTION_NAME}")
            except Exception as e2:
                logger.error(f"Failed to create ChromaDB collection: {str(e2)}")
                vector_db_available = False
                return False
        
        vector_db_available = True
        return True
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}", exc_info=True)
        vector_db_available = False
        return False

def get_collection_stats():
    """Get statistics about the vector database collection"""
    if not vector_db_available:
        return {
            "status": "unavailable",
            "message": "Vector database is not available",
            "error": "ChromaDB could not be initialized"
        }
    
    if not collection:
        return {
            "status": "not_initialized",
            "message": "Vector database collection not initialized",
            "collection_name": CHROMA_COLLECTION_NAME,
            "persist_directory": CHROMA_PERSIST_DIRECTORY
        }
    
    try:
        count = collection.count()
        return {
            "status": "available",
            "collection_name": CHROMA_COLLECTION_NAME,
            "vector_count": count,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "persist_directory": CHROMA_PERSIST_DIRECTORY
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting collection stats: {str(e)}",
            "collection_name": CHROMA_COLLECTION_NAME
        }

async def store_embedding(
    image_id: uuid.UUID,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> bool:
    """Store an image embedding in the vector database"""
    if not vector_db_available or not collection:
        logger.warning("Cannot store embedding: Vector database not available")
        return False
    
    try:
        # Prepare metadata - ChromaDB requires string values for all metadata fields
        doc_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                doc_metadata[key] = str(value)
            elif value is None:
                doc_metadata[key] = ""
            else:
                doc_metadata[key] = str(value)
        
        # Store in ChromaDB
        collection.add(
            ids=[str(image_id)],
            embeddings=[embedding],
            metadatas=[doc_metadata],
            documents=[f"Image: {metadata.get('original_filename', '')}"]
        )
        
        logger.info(f"Successfully stored embedding for image {image_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error storing embedding: {str(e)}")
        return False

async def search_by_embedding(
    query_embedding: List[float],
    team_id: Optional[uuid.UUID] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for similar images based on embedding vector"""
    if not vector_db_available or not collection:
        logger.warning("Cannot perform search: Vector database not available")
        return []
    
    try:
        # Prepare query parameters
        where_clause = {}
        if team_id:
            where_clause = {"team_id": str(team_id)}
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause if where_clause else None
        )
        
        # Process results
        processed_results = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                processed_results.append({
                    "image_id": uuid.UUID(doc_id),
                    "similarity_score": float(results["distances"][0][i]) if "distances" in results and results["distances"] else 0.0,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {}
                })
        
        logger.info(f"Found {len(processed_results)} similar images")
        return processed_results
    
    except Exception as e:
        logger.error(f"Error searching by embedding: {str(e)}")
        return []

def get_all_embeddings(team_id: Optional[uuid.UUID] = None, limit: int = 1000) -> Dict[str, Any]:
    """Get all embeddings for visualization or analysis"""
    if not vector_db_available or not collection:
        logger.warning("Cannot get embeddings: Vector database not available")
        return {"embeddings": [], "metadata": []}
    
    try:
        # Prepare query parameters
        where_clause = {}
        if team_id:
            where_clause = {"team_id": str(team_id)}
        
        # Query all embeddings
        results = collection.query(
            query_embeddings=None,
            n_results=limit,
            where=where_clause if where_clause else None,
            include=["embeddings", "metadatas", "documents"]
        )
        
        return {
            "ids": results.get("ids", [[]]),
            "embeddings": results.get("embeddings", [[]]),
            "metadata": results.get("metadatas", [[]]),
            "documents": results.get("documents", [[]])
        }
    except Exception as e:
        logger.error(f"Error getting all embeddings: {str(e)}")
        return {"embeddings": [], "metadata": []}

# Try to initialize on module import, but don't fail if it doesn't work
try:
    init_vector_db()
except Exception as e:
    logger.error(f"Failed to initialize vector database on import: {str(e)}")
    logger.warning("Vector database features will be disabled")
    vector_db_available = False

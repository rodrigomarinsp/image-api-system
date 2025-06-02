"""
database.py - Database connection and models

This module handles database connection configuration, SQLAlchemy models,
and core database utilities.
"""
import uuid
import logging
import asyncio  # Adicione esta importação
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Integer, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@db:5432/image_api")

# SQLAlchemy setup
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Utility to get database session
async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

# Database models
class Team(Base):
    __tablename__ = "teams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="team")
    images = relationship("Image", back_populates="team")


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, nullable=False, unique=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class Image(Base):
    __tablename__ = "images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    original_filename = Column(String, nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    uploaded_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # Renomeado de "metadata" para "image_metadata" para evitar conflito com o atributo reservado
    image_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="images")
    embedding = relationship("ImageEmbedding", uselist=False, back_populates="image")


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, unique=True)
    embedding_vector = Column(JSONB, nullable=False)  # Store as JSON for simplicity
    embedding_model = Column(String, nullable=False)
    vector_db_id = Column(String, nullable=True)  # ID in the vector database
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    image = relationship("Image", back_populates="embedding")


class AccessLog(Base):
    __tablename__ = "access_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)
    ip_address = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    details = Column(JSONB, nullable=True)


# Função para esperar que o banco de dados esteja disponível
async def wait_for_db(max_retries=10, retry_interval=5):
    """Wait for database to be ready"""
    logger.info("Waiting for database to be ready...")
    retries = 0
    
    while retries < max_retries:
        try:
            # Tenta criar uma engine e conectar
            temp_engine = create_async_engine(DATABASE_URL, echo=False)
            async with temp_engine.begin() as conn:
                # Use text() para executar queries SQL como texto
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))  # Corrigido: use text() para SQL literal

            logger.info("Database is ready!")
            return True
        except Exception as e:
            retries += 1
            logger.warning(f"Connection attempt {retries}/{max_retries} failed: {str(e)}")
            if retries < max_retries:
                logger.info(f"Retrying in {retry_interval} seconds...")
                await asyncio.sleep(retry_interval)
    
    logger.error(f"is not possible to connect to the database after {max_retries} tries")
    return False


# Database initialization function
async def init_db():
    """Initialize database connection and create tables if they don't exist"""
    # Wait for the database to be ready
    if not await wait_for_db():
        raise Exception("is not possible to connect to the database after multiple attempts")
    
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized")

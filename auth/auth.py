"""
This module handles API key authentication and access control
"""
import uuid
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import Request, Response, Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader, APIKeyQuery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Import necessary components from database
from db.database import get_db, APIKey, User, Team, AccessLog

# Configure logging
logger = logging.getLogger(__name__)

# Security schemas
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# API Key generation
def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"imapi_{secrets.token_urlsafe(32)}"


async def get_current_user(
    api_key_header: str = Depends(api_key_header),
    api_key_query: str = Depends(api_key_query),
    x_mock_user: Optional[str] = Header(None),  # Novo parâmetro para receber mock user
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get the current authenticated user.
    Can use a custom mock user configuration passed via X-Mock-User header.
    """
    # Check if a custom mock user was provided in header
    if x_mock_user:
        try:
            # Decode the base64 mock user configuration
            mock_config = json.loads(base64.b64decode(x_mock_user).decode())
            
            # Create a mock User with the provided configuration
            class CustomMockUser:
                def __init__(self):
                    self.id = uuid.UUID(mock_config.get("user_id", "f7beca17-bb9a-4027-929b-b642ca19542b"))
                    self.username = mock_config.get("username", "direct_user")
                    self.email = mock_config.get("email", "direct@example.com")
                    self.team_id = uuid.UUID(mock_config.get("team_id", "11111111-1111-1111-1111-111111111111"))
                    self.is_admin = mock_config.get("is_admin", True)
                    self.created_at = datetime.now(timezone.utc)
                    self.updated_at = None
                    
                def __dict__(self):
                    return {
                        "id": self.id,
                        "username": self.username,
                        "email": self.email,
                        "team_id": self.team_id,
                        "is_admin": self.is_admin,
                        "created_at": self.created_at,
                        "updated_at": self.updated_at
                    }
                    
                def dict(self):
                    return self.__dict__()
            
            logger.info(f"Using custom mock user: {mock_config.get('username')} with team {mock_config.get('team_id')}")
            return CustomMockUser()
        except Exception as e:
            logger.error(f"Error parsing custom mock user: {str(e)}")
            # Fall through to default mock user
    
    # Original mock user code
    class MockUser:
        def __init__(self):
            self.id = uuid.UUID("8c80e144-9f40-4659-92c0-567a6a648a17")
            self.username = "TELLME_DIRECT"
            self.email = "direct@example.com"
            self.team_id = uuid.UUID("ee6a02e1-fa41-4bad-9b5c-afbf69ce59a8")
            self.is_admin = True
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = None
            
        def __dict__(self):
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "team_id": self.team_id,
                "is_admin": self.is_admin,
                "created_at": self.created_at,
                "updated_at": self.updated_at
            }
            
        def dict(self):
            return self.__dict__()
            
    # Return the default mock user
    logger.warning("Authentication disabled: Using default admin user")
    return MockUser()

# Function to verify team access permissions
async def verify_team_access(user: User, team_id: uuid.UUID) -> bool:
    """ Check if the user has access to the specified team.
    This function checks if the user is part of the team or is an admin.
    AUTHENTICATION DISABLED: Always returns True
    """
    # Authentication disabled: always allow access
    return True
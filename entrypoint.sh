#!/bin/bash

set -e

echo "============================================"
echo "Starting Image Management API initialization"
echo "============================================"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h db -U postgres; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done
echo "PostgreSQL is up and running!"

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
for i in {1..30}; do
  if curl -s http://minio:9000/minio/health/live > /dev/null; then
    echo "MinIO is up and running!"
    break
  fi
  echo "Waiting for MinIO..."
  sleep 1
  if [ $i -eq 30 ]; then
    echo "Warning: MinIO may not be ready, but continuing..."
  fi
done

# Check if alembic.ini exists before running migrations
if [ -f "alembic.ini" ]; then
  echo "Running database migrations with Alembic..."
  python -m alembic upgrade head
else
  echo "Alembic configuration not found. Creating database tables directly using SQLAlchemy..."
  # Run a Python script to create tables directly with SQLAlchemy
  python -c "
import asyncio
import os
import sys
sys.path.append('/app')

from db.database import Base, engine

async def create_tables():
    print('Creating database tables...')
    async with engine.begin() as conn:
        # Create all tables defined in SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)
        print('Database tables created successfully.')

# Run the async function
asyncio.run(create_tables())
"
fi

# Create admin team, admin user, and API key
echo "Creating admin team, admin user, and API key if they don't exist..."
python -c "
import asyncio
import uuid
import os
import sys
from datetime import datetime, timedelta

sys.path.append('/app')

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

# Import models
try:
    from db.database import Team, User, APIKey
except ImportError:
    print('Error importing models. Make sure database.py is properly set up.')
    sys.exit(1)

# Predefined IDs for consistency (prevent duplicate creation)
ADMIN_TEAM_ID = uuid.UUID('11111111-1111-1111-1111-111111111111')
ADMIN_USER_ID = uuid.UUID('f7beca17-bb9a-4027-929b-b642ca19542b')  # Original default admin ID
ADMIN_KEY_ID = uuid.UUID('22222222-2222-2222-2222-222222222222')
ADMIN_API_KEY = 'imapi_admin_key_for_secure_authenticated_access'

async def create_admin_entities():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('DATABASE_URL environment variable not set')
        return
        
    print(f'Connecting to database: {db_url}')
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Check if admin team exists
        result = await session.execute(select(Team).where(Team.id == ADMIN_TEAM_ID))
        admin_team = result.scalar_one_or_none()
        
        if not admin_team:
            print('Creating admin team...')
            admin_team = Team(
                id=ADMIN_TEAM_ID,
                name='Admin Team',
                description='Team for system administrators'
            )
            session.add(admin_team)
            await session.commit()
            await session.refresh(admin_team)
            print(f'Admin team created with ID: {admin_team.id}')
        else:
            print(f'Admin team already exists with ID: {admin_team.id}')
        
        # Check if admin user exists
        result = await session.execute(select(User).where(User.id == ADMIN_USER_ID))
        admin_user = result.scalar_one_or_none()
        
        if not admin_user:
            print('Creating admin user...')
            admin_user = User(
                id=ADMIN_USER_ID,
                username='admin',
                email='admin@example.com',
                team_id=admin_team.id,
                is_admin=True
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)
            print(f'Admin user created with ID: {admin_user.id}')
        else:
            # Update admin user's team_id if it's not set correctly
            if admin_user.team_id != admin_team.id:
                print(f'Updating admin user team_id from {admin_user.team_id} to {admin_team.id}')
                admin_user.team_id = admin_team.id
                await session.commit()
                await session.refresh(admin_user)
            print(f'Admin user already exists with ID: {admin_user.id}')
        
        # Check if admin API key exists
        result = await session.execute(select(APIKey).where(APIKey.user_id == admin_user.id))
        admin_key = result.scalar_one_or_none()
        
        if not admin_key:
            print('Creating admin API key...')
            admin_key = APIKey(
                id=ADMIN_KEY_ID,
                key=ADMIN_API_KEY,
                name='Admin API Key',
                user_id=admin_user.id,
                is_active=True,
                expires_at=datetime.now() + timedelta(days=365)  # Valid for 1 year
            )
            session.add(admin_key)
            await session.commit()
            print(f'Admin API key created with ID: {admin_key.id}')
            print(f'Admin API key value: {admin_key.key}')
        else:
            print(f'Admin API key already exists with ID: {admin_key.id}')
            print(f'Admin API key value: {admin_key.key}')
        
    # Close the engine connection
    await engine.dispose()
    print('Admin setup completed successfully')

# Run the async function
try:
    asyncio.run(create_admin_entities())
except Exception as e:
    print(f'Error creating admin entities: {str(e)}')
"

# Create necessary directories
echo "Creating required directories..."
mkdir -p /app/data/chroma_db

# Set proper permissions
echo "Setting file permissions..."
chown -R 1000:1000 /app/data 2>/dev/null || echo "Warning: Could not set permissions on /app/data"

echo "Initialization complete!"
echo "============================================"
echo "Starting Image Management API"
echo "============================================"

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port 8080

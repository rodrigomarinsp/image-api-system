#!/bin/bash
# Advanced deployment script for Image API with Cloud Run support
# Handles local deployment, cloud deployment with proper image tagging

# Global configuration
PROJECT_ROOT=$(pwd)
GCP_PROJECT="image-management-api"
GCP_REGION="europe-west3"
PROJECT_PREFIX="image-api"
DEBUG="true"  # Set to true for detailed debug output

# Function to display usage
usage() {
  echo "Usage: $0 [local|cloud|destroy]"
  echo "  local   - Run application locally using Docker Compose"
  echo "  cloud   - Deploy application to Google Cloud Run"
  echo "  destroy - Destroy all infrastructure created in the cloud"
  exit 1
}

# Function to display errors and exit
error_exit() {
  echo "ERROR: $1" >&2
  echo "Operation failed. Please check the logs above for details."
  exit 1
}

# Function to display informational messages
info() {
  echo "INFO: $1"
}

# Function to display debug messages if DEBUG is enabled
debug() {
  if [ "$DEBUG" = "true" ]; then
    echo "DEBUG: $1"
  fi
}

# Function to check command result
check_result() {
  if [ $? -ne 0 ]; then
    error_exit "$1"
  fi
}

# Verify input parameters
if [ $# -ne 1 ]; then
  usage
fi

ENVIRONMENT=$1
BUILD_TAG=$(date +%Y%m%d%H%M%S)

# Add git hash if available
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
  GIT_HASH=$(git rev-parse --short HEAD)
  BUILD_TAG="${BUILD_TAG}-${GIT_HASH}"
fi

# Find project files in any structure
find_project_structure() {
  info "Detecting project structure..."
  
  # Debug: List files in current directory
  debug "Current directory: $(pwd)"
  debug "Files in current directory:"
  ls -la
  
  # Look for app directory
  APP_DIR=""
  MAIN_PATH=""
  
  # Check if app directory exists directly or in known subdirectories
  potential_app_dirs=(
    "./app"
    "./image-api/app"
    "../app"
  )
  
  for dir in "${potential_app_dirs[@]}"; do
    if [ -d "$dir" ]; then
      APP_DIR=$(realpath "$dir")
      debug "Found app directory at: $APP_DIR"
      break
    fi
  done
  
  # If app directory not found, look for main.py or main.txt
  if [ -z "$APP_DIR" ]; then
    debug "App directory not found, looking for main.py or main.txt..."
    
    potential_main_files=(
      "./main.py"
      "./main.txt"
      "./image-api/main.py"
      "./image-api/main.txt"
    )
    
    for main_file in "${potential_main_files[@]}"; do
      if [ -f "$main_file" ]; then
        MAIN_PATH=$(realpath "$main_file")
        debug "Found main file at: $MAIN_PATH"
        break
      fi
    done
    
    # If still not found, do recursive search
    if [ -z "$MAIN_PATH" ]; then
      MAIN_PATH=$(find "$PROJECT_ROOT" -name "main.py" -type f | head -n 1)
      if [ -z "$MAIN_PATH" ]; then
        MAIN_PATH=$(find "$PROJECT_ROOT" -name "main.txt" -type f | head -n 1)
      fi
      
      if [ -n "$MAIN_PATH" ]; then
        debug "Found main file through recursive search: $MAIN_PATH"
      fi
    fi
  fi
  
  # Make sure we found either app dir or main path
  if [ -z "$APP_DIR" ] && [ -z "$MAIN_PATH" ]; then
    error_exit "Could not find Python application files."
  fi
  
  # Find entrypoint.sh
  ENTRYPOINT_PATH=""
  potential_entrypoint_paths=(
    "./entrypoint.sh"
    "./entrypoint.txt"
    "./deploy/docker/entrypoint.sh"
    "./image-api/deploy/docker/entrypoint.sh"
  )
  
  for path in "${potential_entrypoint_paths[@]}"; do
    if [ -f "$path" ]; then
      ENTRYPOINT_PATH=$(realpath "$path")
      debug "Found entrypoint script at: $ENTRYPOINT_PATH"
      break
    fi
  done
  
  # If not found, do recursive search
  if [ -z "$ENTRYPOINT_PATH" ]; then
    ENTRYPOINT_PATH=$(find "$PROJECT_ROOT" -name "entrypoint.sh" -type f | head -n 1)
    if [ -z "$ENTRYPOINT_PATH" ]; then
      ENTRYPOINT_PATH=$(find "$PROJECT_ROOT" -name "entrypoint.txt" -type f | head -n 1)
    fi
    
    if [ -n "$ENTRYPOINT_PATH" ]; then
      debug "Found entrypoint script through recursive search: $ENTRYPOINT_PATH"
    else
      debug "No entrypoint script found, will use direct command instead."
    fi
  fi
  
  # Find requirements.txt
  REQUIREMENTS_PATH=""
  potential_requirements_paths=(
    "./requirements.txt"
    "./image-api/requirements.txt"
  )
  
  for path in "${potential_requirements_paths[@]}"; do
    if [ -f "$path" ]; then
      REQUIREMENTS_PATH=$(realpath "$path")
      debug "Found requirements.txt at: $REQUIREMENTS_PATH"
      break
    fi
  done
  
  # If not found, do recursive search
  if [ -z "$REQUIREMENTS_PATH" ]; then
    REQUIREMENTS_PATH=$(find "$PROJECT_ROOT" -name "requirements.txt" -type f | head -n 1)
    
    if [ -n "$REQUIREMENTS_PATH" ]; then
      debug "Found requirements.txt through recursive search: $REQUIREMENTS_PATH"
    else
      error_exit "Could not find requirements.txt."
    fi
  fi
  
  # Find terraform-admin-key.json
  TERRAFORM_ADMIN_KEY_PATH=""
  potential_key_paths=(
    "./deploy/terraform/service-accounts/terraform-admin-key.json"
    "./terraform-admin-key.json"
    "./terraform-admin-key.json.txt"
    "./image-api/deploy/terraform/service-accounts/terraform-admin-key.json"
  )
  
  for path in "${potential_key_paths[@]}"; do
    if [ -f "$path" ]; then
      TERRAFORM_ADMIN_KEY_PATH=$(realpath "$path")
      debug "Found Terraform admin key at: $TERRAFORM_ADMIN_KEY_PATH"
      break
    fi
  done
  
  # If not found, do recursive search
  if [ -z "$TERRAFORM_ADMIN_KEY_PATH" ]; then
    TERRAFORM_ADMIN_KEY_PATH=$(find "$PROJECT_ROOT" -name "terraform-admin-key.json" -type f | head -n 1)
    if [ -z "$TERRAFORM_ADMIN_KEY_PATH" ]; then
      TERRAFORM_ADMIN_KEY_PATH=$(find "$PROJECT_ROOT" -name "terraform-admin-key.json.txt" -type f | head -n 1)
    fi
    
    if [ -n "$TERRAFORM_ADMIN_KEY_PATH" ]; then
      debug "Found Terraform admin key through recursive search: $TERRAFORM_ADMIN_KEY_PATH"
    else
      error_exit "Could not find Terraform admin key."
    fi
  fi
  
  # Find Terraform directory
  TERRAFORM_DIR=""
  potential_terraform_dirs=(
    "./deploy/terraform"
    "./terraform"
    "./image-api/deploy/terraform"
  )
  
  for dir in "${potential_terraform_dirs[@]}"; do
    if [ -d "$dir" ]; then
      TERRAFORM_DIR=$(realpath "$dir")
      debug "Found Terraform directory at: $TERRAFORM_DIR"
      break
    fi
  done
  
  # If not found, look for main.tf
  if [ -z "$TERRAFORM_DIR" ]; then
    MAIN_TF_PATH=$(find "$PROJECT_ROOT" -name "main.tf" -type f | head -n 1)
    if [ -n "$MAIN_TF_PATH" ]; then
      TERRAFORM_DIR=$(dirname "$MAIN_TF_PATH")
      debug "Found Terraform directory through main.tf: $TERRAFORM_DIR"
    else
      # Create Terraform directory
      TERRAFORM_DIR="${PROJECT_ROOT}/terraform"
      mkdir -p "$TERRAFORM_DIR"
      debug "Created new Terraform directory at: $TERRAFORM_DIR"
    fi
  fi
  
  debug "Project structure detection complete."
  debug "APP_DIR: $APP_DIR"
  debug "MAIN_PATH: $MAIN_PATH"
  debug "ENTRYPOINT_PATH: $ENTRYPOINT_PATH"
  debug "REQUIREMENTS_PATH: $REQUIREMENTS_PATH"
  debug "TERRAFORM_ADMIN_KEY_PATH: $TERRAFORM_ADMIN_KEY_PATH"
  debug "TERRAFORM_DIR: $TERRAFORM_DIR"
}

# Check if Docker is available and working
check_docker() {
  info "Checking if Docker is available..."
  if ! command -v docker &> /dev/null; then
    info "Docker command not found."
    return 1
  fi
  
  # Check if Docker daemon is running
  if ! docker info &> /dev/null; then
    info "Docker daemon is not running or not accessible."
    return 1
  fi
  
  info "Docker is available and working."
  return 0
}

# Test connection to Google Cloud
test_gcp_connection() {
  echo "Testing Google Cloud connection..."
  gcloud projects describe $GCP_PROJECT --quiet > /dev/null 2>&1
  check_result "Failed to connect to Google Cloud. Check your credentials and network."
  echo "Google Cloud connection successful."
}

# Authenticate with Google Cloud
authenticate_gcp() {
  if [ -f "$TERRAFORM_ADMIN_KEY_PATH" ]; then
    echo "Authenticating with Google Cloud using Terraform Admin key..."
    export GOOGLE_APPLICATION_CREDENTIALS="$TERRAFORM_ADMIN_KEY_PATH"
    
    # Extract service account email
    TERRAFORM_SA_EMAIL=$(cat "$TERRAFORM_ADMIN_KEY_PATH" | jq -r '.client_email')
    
    if [ -n "$TERRAFORM_SA_EMAIL" ] && [ "$TERRAFORM_SA_EMAIL" != "null" ]; then
      echo "Using service account: $TERRAFORM_SA_EMAIL"
      gcloud auth activate-service-account "$TERRAFORM_SA_EMAIL" --key-file="$TERRAFORM_ADMIN_KEY_PATH" --project="$GCP_PROJECT" --quiet
      check_result "Failed to authenticate with Google Cloud"
    else
      error_exit "Could not determine service account email from terraform admin key"
    fi
  else
    error_exit "Terraform admin key not found at $TERRAFORM_ADMIN_KEY_PATH"
  fi
  
  gcloud config set project "$GCP_PROJECT"
  gcloud config set compute/region "$GCP_REGION"
}

# Enable required APIs for GCP
enable_required_apis() {
  echo "Checking and enabling required APIs..."
  
  # List of required APIs
  REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "artifactregistry.googleapis.com"
    "run.googleapis.com"
    "secretmanager.googleapis.com"
    "iam.googleapis.com"
    "storage.googleapis.com"
  )
  
  for api in "${REQUIRED_APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api --quiet
    check_result "Failed to enable $api"
  done
}

# Prepare build directory with optimized Dockerfile for Cloud Run
prepare_build_directory() {
  local build_dir="$1"
  
  info "Preparing build directory..."
  if [ -d "$build_dir" ]; then
    rm -rf "$build_dir"
  fi
  mkdir -p "$build_dir"
  check_result "Failed to create build directory"
  
  info "Copying necessary files..."
  
  # Create app directory in build_dir
  mkdir -p "$build_dir/app"
  
  # Copy application files
  if [ -n "$APP_DIR" ] && [ -d "$APP_DIR" ]; then
    debug "Copying from APP_DIR: $APP_DIR"
    cp -r "$APP_DIR"/* "$build_dir/app/"
    check_result "Failed to copy app directory contents"
  elif [ -n "$MAIN_PATH" ]; then
    debug "Using MAIN_PATH: $MAIN_PATH"
    
    # Copy Python files as needed
    if [[ "$MAIN_PATH" == *.txt ]]; then
      cp "$MAIN_PATH" "$build_dir/app/main.py"
    else
      cp "$MAIN_PATH" "$build_dir/app/"
    fi
    
    # Copy other relevant Python files
    main_dir=$(dirname "$MAIN_PATH")
    for file in "$main_dir"/*.{py,txt}; do
      if [ -f "$file" ] && [ "$(basename "$file")" != "$(basename "$MAIN_PATH")" ]; then
        file_name=$(basename "$file")
        
        # Convert .txt to .py when appropriate
        if [[ "$file_name" == *.txt ]] && [[ "$file_name" != "requirements.txt" ]] && \
           [[ "$file_name" != "Dockerfile.txt" ]] && [[ "$file_name" != "docker-compose.txt" ]]; then
          module_name="${file_name%.txt}.py"
          cp "$file" "$build_dir/app/$module_name"
        elif [[ "$file_name" == *.py ]]; then
          cp "$file" "$build_dir/app/"
        fi
      fi
    done
  fi
  
  # Verify application files were copied correctly
  if [ ! "$(ls -A $build_dir/app/ 2>/dev/null)" ]; then
    error_exit "Failed to copy application files to build directory"
  fi
  
  # Copy requirements.txt
  cp "$REQUIREMENTS_PATH" "$build_dir/"
  
  # Create optimized Dockerfile for Cloud Run
  cat > "$build_dir/Dockerfile" << 'EOF'
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install PostgreSQL client tools for initialization script
RUN apt-get update && apt-get install -y postgresql-client curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for initialization scripts
RUN mkdir -p /app/scripts

# Copy application files
COPY . .

# Create startup script
# COPY entrypoint.sh .
# RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 8080
EOF
  
  # Create .gcloudignore
  cat > "$build_dir/.gcloudignore" << 'EOF'
.git
.gitignore
.gcloudignore
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
venv/
.env
*.log
EOF

  debug "Build directory contents:"
  ls -la "$build_dir"
  debug "app directory contents:"
  ls -la "$build_dir/app" || echo "Could not list app directory"
  debug "Dockerfile contents:"
  cat "$build_dir/Dockerfile"

  return 0
}

# Check if image exists in Artifact Registry
check_image_exists() {
  local repo_path="$1"
  local tag="${2:-latest}"
  
  info "Checking if image ${repo_path}:${tag} exists in Artifact Registry..."
  gcloud artifacts docker images describe "${repo_path}:${tag}" --quiet &> /dev/null
  return $?
}

# Build Docker image locally
build_docker_image_locally() {
  local build_dir="$1"
  local repo_path="$2"
  local image_tag="$3"

  info "Building Docker image locally with platform targeting..."
  
  if ! check_docker; then
    info "Docker is not available for local build."
    return 1
  fi
  
  cd "$build_dir"
  # Use --platform=linux/amd64 to build for Cloud Run compatibility
  docker build --platform=linux/amd64 -t "${repo_path}:${image_tag}" -t "${repo_path}:latest" .
  if [ $? -ne 0 ]; then
    info "Docker build failed locally."
    return 1
  fi
  
  info "Pushing image to Artifact Registry..."
  docker push "${repo_path}:${image_tag}" && docker push "${repo_path}:latest"
  if [ $? -ne 0 ]; then
    info "Failed to push image to Artifact Registry."
    return 1
  fi
  
  info "Image built and pushed successfully via local Docker."
  return 0
}

# Build image using Cloud Build
build_docker_image_cloud() {
  local build_dir="$1"
  local repo_path="$2"
  local image_tag="$3"
  local cloud_build_config="${build_dir}/cloudbuild.yaml"
  
  info "Building image using Cloud Build..."
  
  # Create Cloud Build configuration
  cat > "$cloud_build_config" << EOF
# Cloud Build configuration for image building
steps:
  # Build the Docker image with platform specification
  - name: 'gcr.io/cloud-builders/docker'
    args: 
      - 'build'
      - '--platform=linux/amd64'
      - '-t'
      - '${repo_path}:${image_tag}'
      - '-t'
      - '${repo_path}:latest'
      - '.'
    id: 'build'

  # Push the Docker image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${repo_path}:${image_tag}']
    id: 'push-version'
    waitFor: ['build']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${repo_path}:latest']
    id: 'push-latest'
    waitFor: ['build']

images:
  - '${repo_path}:${image_tag}'
  - '${repo_path}:latest'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'
EOF
  
  cd "$build_dir"
  gcloud builds submit --config="$cloud_build_config" . --timeout=30m
  if [ $? -ne 0 ]; then
    info "Cloud Build failed."
    return 1
  fi
  
  info "Image built and pushed successfully via Cloud Build."
  return 0

}

# Deploy to Cloud Run with explicit tag
deploy_to_cloud_run() {
  local image="$1"
  local service_name="$2"
  local region="$3"
  local service_account="$4"
  local storage_bucket="$5"
  
  info "Verifying image exists before deployment: $image"
  
  # Extract repository and tag from the image
  local repo_part=${image%:*}
  local tag_part=${image#*:}
  
  if [ "$repo_part" = "$tag_part" ]; then
    # No tag was specified, use latest
    tag_part="latest"
    image="${image}:latest"
  fi
  
  # Verify image exists in Artifact Registry
  if ! gcloud artifacts docker images describe "$image" --quiet &> /dev/null; then
    error_exit "Image $image does not exist in Artifact Registry. Please build it first."
  fi
  
  info "Deploying image to Cloud Run: $image"
  
  # Deploy with minimal configuration
  gcloud run deploy "$service_name" \
    --image="$image" \
    --region="$region" \
    --platform=managed \
    --service-account="$service_account" \
    --set-env-vars="DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/image_api" \
    --set-env-vars="GCS_BUCKET_NAME=${storage_bucket}" \
    --set-env-vars="USE_MINIO=false" \
    --set-env-vars="CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db" \
    --set-env-vars="CHROMA_COLLECTION_NAME=image_embeddings" \
    --set-env-vars="EMBEDDING_MODEL=clip" \
    --set-env-vars="EMBEDDING_DIMENSION=512" \
    --memory=1Gi \
    --cpu=1 \
    --allow-unauthenticated
    
  local deploy_result=$?
  
  if [ $deploy_result -ne 0 ]; then
    info "Deployment failed."
    return 1
  fi
  
  # Get service URL
  local service_url=""
  service_url=$(gcloud run services describe "$service_name" \
    --region="$region" \
    --format="value(status.url)" 2>/dev/null)
  
  if [ -n "$service_url" ]; then
    info "Service deployed successfully at: $service_url"
    echo "$service_url" > service_url.txt
    return 0
  else
    info "Service URL not available."
    return 1
  fi
}

# Create Terraform files needed for infrastructure
create_terraform_files() {
  local terraform_dir="$1"
  
  if [ ! -f "${terraform_dir}/main.tf" ]; then
    info "Creating main.tf..."
    cat > "${terraform_dir}/main.tf" << EOF
# main.tf - Infrastructure definition for Image API on GCP

provider "google" {
  project     = var.gcp_project_id
  region      = var.gcp_region
  credentials = file("\${path.module}/service-accounts/terraform-admin-key.json")
}

# Google Cloud Storage bucket for image storage
resource "google_storage_bucket" "image_storage" {
  name          = "\${var.project_prefix}-images"
  location      = var.gcp_region
  storage_class = "STANDARD"
  
  uniform_bucket_level_access = true
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Service account for the Cloud Run service
resource "google_service_account" "cloudrun_service_account" {
  account_id   = "\${var.project_prefix}-sa"
  display_name = "Service Account for Image API Cloud Run service"
}

# Grant storage admin role to service account
resource "google_project_iam_member" "storage_admin" {
  project = var.gcp_project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:\${google_service_account.cloudrun_service_account.email}"
}

# Secret Manager for storing sensitive configuration
resource "google_secret_manager_secret" "db_password" {
  secret_id = "\${var.project_prefix}-db-password"
  
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret       = google_secret_manager_secret.db_password.id
  secret_data  = var.db_password
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "\${var.project_prefix}-api-key"
  
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "api_key_version" {
  secret       = google_secret_manager_secret.api_key.id
  secret_data  = var.api_key
}

# Grant secret accessor role to service account
resource "google_secret_manager_secret_iam_member" "db_password_accessor" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:\${google_service_account.cloudrun_service_account.email}"
}

resource "google_secret_manager_secret_iam_member" "api_key_accessor" {
  secret_id = google_secret_manager_secret.api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:\${google_service_account.cloudrun_service_account.email}"
}

# Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "image_repo" {
  location      = var.gcp_region
  repository_id = "\${var.project_prefix}-repo"
  format        = "DOCKER"
  
  depends_on = [google_project_iam_member.storage_admin]
}
EOF
  fi

  if [ ! -f "${terraform_dir}/variables.tf" ]; then
    info "Creating variables.tf..."
    cat > "${terraform_dir}/variables.tf" << EOF
# variables.tf - Variables for Terraform configuration

variable "gcp_project_id" {
  description = "The Google Cloud Project ID"
  type        = string
}

variable "gcp_region" {
  description = "The Google Cloud region where resources will be created"
  type        = string
  default     = "europe-west3" # Frankfurt
}

variable "project_prefix" {
  description = "Prefix for naming resources"
  type        = string
  default     = "image-api"
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

variable "api_key" {
  description = "Default API key for initial setup"
  type        = string
  sensitive   = true
}

variable "database_url" {
  description = "PostgreSQL database URL"
  type        = string
  default     = "postgresql+asyncpg://postgres:postgres@db:5432/image_api"
}
EOF
  fi

  if [ ! -f "${terraform_dir}/outputs.tf" ]; then
    info "Creating outputs.tf..."
    cat > "${terraform_dir}/outputs.tf" << EOF
# outputs.tf - Output values from Terraform

output "storage_bucket" {
  description = "Cloud Storage bucket for images"
  value       = google_storage_bucket.image_storage.name
}

output "service_account_email" {
  description = "Service Account email"
  value       = google_service_account.cloudrun_service_account.email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository for Docker images"
  value       = "\${var.gcp_region}-docker.pkg.dev/\${var.gcp_project_id}/\${google_artifact_registry_repository.image_repo.repository_id}"
}
EOF
  fi

  # Ensure service-accounts directory exists
  mkdir -p "${terraform_dir}/service-accounts"
  
  # If key file is elsewhere, copy it to the service-accounts directory
  if [ ! -f "${terraform_dir}/service-accounts/terraform-admin-key.json" ] && [ -f "$TERRAFORM_ADMIN_KEY_PATH" ]; then
    info "Copying terraform-admin-key.json to service-accounts directory..."
    if [[ "$TERRAFORM_ADMIN_KEY_PATH" == *.txt ]]; then
      cp "$TERRAFORM_ADMIN_KEY_PATH" "${terraform_dir}/service-accounts/terraform-admin-key.json"
      debug "Copied $TERRAFORM_ADMIN_KEY_PATH to ${terraform_dir}/service-accounts/terraform-admin-key.json with extension change"
    else
      cp "$TERRAFORM_ADMIN_KEY_PATH" "${terraform_dir}/service-accounts/"
      debug "Copied $TERRAFORM_ADMIN_KEY_PATH to ${terraform_dir}/service-accounts/ directly"
    fi
  fi

  return 0
}

# Deploy locally
deploy_local() {
  echo "========================================="
  echo "Deploying Image API locally using Docker Compose"
  echo "========================================="

  # Check if Docker is available
  if ! check_docker; then
    error_exit "Docker is required for local deployment."
  fi

  # Find docker-compose.yml
  local docker_compose_path=""
  if [ -f "docker-compose.yml" ]; then
    docker_compose_path="docker-compose.yml"
  elif [ -f "docker-compose.yaml" ]; then
    docker_compose_path="docker-compose.yaml"
  elif [ -f "docker-compose.txt" ]; then
    # Convert docker-compose.txt to docker-compose.yml
    cp docker-compose.txt docker-compose.yml
    docker_compose_path="docker-compose.yml"
  else
    error_exit "Could not find docker-compose file."
  fi

  # Build and start containers
  docker-compose -f "$docker_compose_path" build
  check_result "Docker build failed"
  
  docker-compose -f "$docker_compose_path" up -d
  check_result "Docker compose up failed"

  echo "========================================="
  echo "Local deployment completed!"
  echo "API is available at: http://localhost:8080"
  echo "========================================="
}

# Deploy to cloud
deploy_cloud() {
  echo "========================================="
  echo "Deploying Image API to Google Cloud Run"
  echo "========================================="

  # Detect project structure
  find_project_structure
  
  # Check for required commands
  for cmd in gcloud jq terraform; do
    if ! command -v $cmd &> /dev/null; then
      error_exit "Required command '$cmd' not found. Please install it and try again."
    fi
  done

  # Authenticate with GCP
  authenticate_gcp
  
  # Test connection to GCP
  test_gcp_connection
  
  # Enable required APIs
  enable_required_apis
  
  # Create/update Terraform files
  create_terraform_files "$TERRAFORM_DIR"
  
  # Prepare and apply Terraform
  cd "$TERRAFORM_DIR"
  
  cat > terraform.tfvars <<EOL
gcp_project_id = "${GCP_PROJECT}"
gcp_region     = "${GCP_REGION}"
project_prefix = "${PROJECT_PREFIX}"
db_password    = "postgres"
api_key        = "initial-api-key"
EOL
  
  # Apply Terraform infrastructure
  echo "Provisioning core infrastructure with Terraform..."
  terraform init -upgrade
  check_result "Terraform initialization failed"
  
  terraform apply -auto-approve
  check_result "Terraform apply failed"
  
  # Get Terraform outputs
  ARTIFACT_REPO=$(terraform output -raw artifact_registry_repository)
  SERVICE_ACCOUNT=$(terraform output -raw service_account_email)
  STORAGE_BUCKET=$(terraform output -raw storage_bucket)
  
  # Configure Cloud Build permissions
  echo "Granting necessary permissions to Cloud Build service account..."
  
  # Get the Cloud Build service account
  CLOUDBUILD_SA="$(gcloud projects describe $GCP_PROJECT --format 'value(projectNumber)')@cloudbuild.gserviceaccount.com"
  
  # Grant permissions to the Cloud Build service account
  gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$CLOUDBUILD_SA" \
    --role="roles/run.admin" --quiet
  check_result "Failed to grant Cloud Run admin role to Cloud Build service account"
  
  gcloud projects add-iam-policy-binding $GCP_PROJECT \
    --member="serviceAccount:$CLOUDBUILD_SA" \
    --role="roles/iam.serviceAccountUser" --quiet
  check_result "Failed to grant Service Account User role to Cloud Build service account"
  
  # Prepare build directory
  BUILD_DIR="${PROJECT_ROOT}/tmp_build"
  prepare_build_directory "$BUILD_DIR"
  check_result "Failed to prepare build directory"
  
  # Full image path with specific tag
  IMAGE_PATH="${ARTIFACT_REPO}/image-api"
  
  # Always build a new image with the current tag
  info "Building new image with tag: ${BUILD_TAG}"
  
  # Try to build locally first
  if build_docker_image_locally "$BUILD_DIR" "$IMAGE_PATH" "$BUILD_TAG"; then
    info "Image built locally and pushed successfully."
    IMAGE_TO_DEPLOY="${IMAGE_PATH}:${BUILD_TAG}"
  else
    info "Local build failed. Attempting to build using Cloud Build..."
    
    # Try to build with Cloud Build
    if build_docker_image_cloud "$BUILD_DIR" "$IMAGE_PATH" "$BUILD_TAG"; then
      info "Image built with Cloud Build and pushed successfully."
      IMAGE_TO_DEPLOY="${IMAGE_PATH}:${BUILD_TAG}"
    else
      error_exit "Failed to build image using both local Docker and Cloud Build."
    fi
  fi
  
  # Explicitly tag the new image as latest
  info "Tagging new image as latest..."
  gcloud artifacts docker tags add "${IMAGE_TO_DEPLOY}" "${IMAGE_PATH}:latest" --quiet
  
  # Deploy to Cloud Run with specific tag (not 'latest')
  if ! deploy_to_cloud_run "$IMAGE_TO_DEPLOY" "${PROJECT_PREFIX}-api" "$GCP_REGION" "$SERVICE_ACCOUNT" "$STORAGE_BUCKET"; then
    error_exit "Failed to deploy to Cloud Run."
  fi
  
  # Get service URL
  SERVICE_URL=$(gcloud run services describe "${PROJECT_PREFIX}-api" --region="$GCP_REGION" --format='value(status.url)')
  
  # Clean up temporary files
  cd "${PROJECT_ROOT}"
  rm -rf "$BUILD_DIR"
  
  echo "========================================="
  echo "Cloud deployment completed successfully!"
  echo "API URL: $SERVICE_URL"
  echo "========================================="
}

# Function to destroy infrastructure
destroy_infrastructure() {
  echo "========================================="
  echo "DESTROYING ALL INFRASTRUCTURE IN THE CLOUD"
  echo "========================================="

    
  # Detect project structure
  find_project_structure
  
  # Check for required commands
  for cmd in gcloud jq terraform; do
    if ! command -v $cmd &> /dev/null; then
      error_exit "Required command '$cmd' not found. Please install it and try again."
    fi
  done
  
  # Confirm destruction
  read -p "Are you sure you want to destroy all infrastructure? This action cannot be undone. (yes/no): " confirm
  if [ "$confirm" != "yes" ]; then
    echo "Destruction aborted."
    exit 0
  fi
  
  # Authenticate with GCP
  authenticate_gcp
  
  # Test connection to GCP
  test_gcp_connection
  
  echo "Starting destruction process..."
  
  # 1. Delete Cloud Run service if it exists
  if gcloud run services describe "${PROJECT_PREFIX}-api" --region="$GCP_REGION" --format="value(name)" &>/dev/null; then
    echo "Deleting Cloud Run service: ${PROJECT_PREFIX}-api"
    gcloud run services delete "${PROJECT_PREFIX}-api" --region="$GCP_REGION" --quiet
    check_result "Failed to delete Cloud Run service"
  else
    echo "Cloud Run service not found, skipping..."
  fi
  
  # 2. Check if we have Terraform files for destruction
  if [ -d "$TERRAFORM_DIR" ]; then
    cd "$TERRAFORM_DIR"
    
    # Check if we need to create terraform.tfvars
    if [ ! -f "terraform.tfvars" ]; then
      cat > terraform.tfvars <<EOL
gcp_project_id = "${GCP_PROJECT}"
gcp_region     = "${GCP_REGION}"
project_prefix = "${PROJECT_PREFIX}"
db_password    = "postgres"
api_key        = "initial-api-key"
EOL
    fi
    
    # Initialize Terraform
    echo "Initializing Terraform..."
    terraform init -upgrade
    
    # Destroy resources with Terraform
    echo "Destroying infrastructure with Terraform..."
    terraform destroy -auto-approve
    
    if [ $? -eq 0 ]; then
      echo "Terraform destroy completed successfully."
    else
      echo "Warning: Terraform destroy may have encountered issues."
      echo "Continuing with manual cleanup of remaining resources..."
    fi
  else
    echo "Terraform directory not found. Will attempt manual resource cleanup..."
  fi
  
  # 3. Clean up images in Artifact Registry
  echo "Checking for images to clean up in Artifact Registry..."
  REPO_PATH="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${PROJECT_PREFIX}-repo"
  
  # List all images and tags
  echo "Listing all images in repository..."
  IMAGES=$(gcloud artifacts docker images list "${REPO_PATH}" --format="value(package)")
  
  if [ -n "$IMAGES" ]; then
    echo "Deleting all images in repository..."
    for img in $IMAGES; do
      echo "Deleting image: $img"
      gcloud artifacts docker images delete "$img" --quiet --delete-tags
    done
  fi
  
  # 4. Delete Artifact Registry repository
  if gcloud artifacts repositories describe "${PROJECT_PREFIX}-repo" --location="$GCP_REGION" &>/dev/null; then
    echo "Deleting Artifact Registry repository: ${PROJECT_PREFIX}-repo"
    gcloud artifacts repositories delete "${PROJECT_PREFIX}-repo" --location="$GCP_REGION" --quiet
  fi
  
  # 5. Clean up Storage bucket
  if gsutil ls "gs://${PROJECT_PREFIX}-images" &>/dev/null; then
    echo "Deleting Storage bucket: ${PROJECT_PREFIX}-images"
    gsutil rm -r "gs://${PROJECT_PREFIX}-images"
  fi
  
  # 6. Clean up Secret Manager secrets
  if gcloud secrets describe "${PROJECT_PREFIX}-db-password" &>/dev/null; then
    echo "Deleting Secret Manager secret: ${PROJECT_PREFIX}-db-password"
    gcloud secrets delete "${PROJECT_PREFIX}-db-password" --quiet
  fi
  
  if gcloud secrets describe "${PROJECT_PREFIX}-api-key" &>/dev/null; then
    echo "Deleting Secret Manager secret: ${PROJECT_PREFIX}-api-key"
    gcloud secrets delete "${PROJECT_PREFIX}-api-key" --quiet
  fi
  
  # 7. Clean up service account
  if gcloud iam service-accounts describe "${PROJECT_PREFIX}-sa@${GCP_PROJECT}.iam.gserviceaccount.com" &>/dev/null; then
    echo "Deleting service account: ${PROJECT_PREFIX}-sa@${GCP_PROJECT}.iam.gserviceaccount.com"
    gcloud iam service-accounts delete "${PROJECT_PREFIX}-sa@${GCP_PROJECT}.iam.gserviceaccount.com" --quiet
  fi
  
  echo "========================================="
  echo "Infrastructure destruction completed!"
  echo "All resources have been removed from Google Cloud."
  echo "========================================="
}

# Execute the appropriate function based on selection
case ${ENVIRONMENT} in
  local)
    deploy_local
    ;;
  cloud)
    deploy_cloud
    ;;
  destroy)
    destroy_infrastructure
    ;;
  *)
    usage
    ;;
esac
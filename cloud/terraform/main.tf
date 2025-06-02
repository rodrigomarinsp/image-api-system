provider "google" {
  project     = var.gcp_project_id
  region      = var.gcp_region
  credentials = file("${path.module}/service-accounts/terraform-admin-key.json")
}

# Google Cloud Storage bucket for image storage
resource "google_storage_bucket" "image_storage" {
  name          = "${var.project_prefix}-images"
  location      = var.gcp_region
  storage_class = "STANDARD"
  force_destroy = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Service account for the Cloud Run service
resource "google_service_account" "cloudrun_service_account" {
  account_id   = "${var.project_prefix}-sa"
  display_name = "Service Account for Image API Cloud Run service"
}

# Grant storage admin role to service account
resource "google_project_iam_member" "storage_admin" {
  project = var.gcp_project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloudrun_service_account.email}"
}

# Grant secretmanager access role to service account
resource "google_project_iam_member" "secretmanager_accessor" {
  project = var.gcp_project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloudrun_service_account.email}"
}

# Secret for database connection
resource "google_secret_manager_secret" "db_connection" {
  secret_id = "${var.project_prefix}-db-connection"
  
  replication {
    user_managed {
      replicas {
        location = var.gcp_region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "db_connection_version" {
  secret      = google_secret_manager_secret.db_connection.id
  secret_data = var.database_url
}

# Secret for MinIO connection
resource "google_secret_manager_secret" "minio_connection" {
  secret_id = "${var.project_prefix}-minio-connection"
  
  replication {
    user_managed {
      replicas {
        location = var.gcp_region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "minio_connection_version" {
  secret      = google_secret_manager_secret.minio_connection.id
  secret_data = "http://minioadmin:minioadmin@minio:9000"
}

# Secret for API key
resource "google_secret_manager_secret" "api_key" {
  secret_id = "${var.project_prefix}-api-key"
  
  replication {
    user_managed {
      replicas {
        location = var.gcp_region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "api_key_version" {
  secret      = google_secret_manager_secret.api_key.id
  secret_data = var.api_key
}

# Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "image_repo" {
  location      = var.gcp_region
  repository_id = "${var.project_prefix}-repo"
  description   = "Docker repository for ${var.project_prefix}"
  format        = "DOCKER"
}

# VPC connector for Cloud Run to connect to VPC resources
resource "google_vpc_access_connector" "connector" {
  name          = "${var.project_prefix}-vpc-connector"
  region        = var.gcp_region
  ip_cidr_range = "10.8.0.0/28"
  network       = "default"
}

# Cloud Run service for the API
resource "google_cloud_run_service" "image_api" {
  name     = "${var.project_prefix}-api"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.image_repo.repository_id}/${var.project_prefix}:latest"
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }

        env {
          name  = "STORAGE_BUCKET"
          value = google_storage_bucket.image_storage.name
        }

        env {
          name  = "DB_CONNECTION_SECRET"
          value = google_secret_manager_secret.db_connection.name
        }

        env {
          name  = "MINIO_CONNECTION_SECRET"
          value = google_secret_manager_secret.minio_connection.name
        }

        env {
          name  = "API_KEY_SECRET"
          value = google_secret_manager_secret.api_key.name
        }
      }

      service_account_name = google_service_account.cloudrun_service_account.email
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale"        = "10"
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-egress"    = "all-traffic"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  # Depend on all secret versions to ensure they exist before deploying
  depends_on = [
    google_secret_manager_secret_version.db_connection_version,
    google_secret_manager_secret_version.minio_connection_version,
    google_secret_manager_secret_version.api_key_version,
    google_artifact_registry_repository.image_repo,
  ]
}

# Allow unauthenticated access to the Cloud Run service
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.image_api.name
  location = google_cloud_run_service.image_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud SQL instance for PostgreSQL
resource "google_sql_database_instance" "postgres" {
  name             = "${var.project_prefix}-db"
  database_version = "POSTGRES_13"
  region           = var.gcp_region
  
  settings {
    tier = "db-f1-micro"  # Smallest tier for development
    
    ip_configuration {
      ipv4_enabled    = true
      private_network = "projects/${var.gcp_project_id}/global/networks/default"
    }
    
    backup_configuration {
      enabled = true
    }
  }
  
  # Prevent destruction by mistake
  deletion_protection = false  # Set to true in production
}

# Create a database
resource "google_sql_database" "database" {
  name     = "image_api"
  instance = google_sql_database_instance.postgres.name
}

# Create a user
resource "google_sql_user" "user" {
  name     = "postgres"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

# Output important values
output "api_url" {
  value = google_cloud_run_service.image_api.status[0].url
}

output "storage_bucket" {
  value = google_storage_bucket.image_storage.name
}

output "service_account_email" {
  value = google_service_account.cloudrun_service_account.email
}

output "artifact_registry_repository" {
  value = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.image_repo.repository_id}"
}

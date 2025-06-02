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
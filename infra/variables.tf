variable "environment" {
  description = "Deployment environment (dev | prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "app_name" {
  description = "Application name used as a prefix for all resources"
  type        = string
  default     = "regllm"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "ecs_cpu" {
  description = "ECS task CPU units (1 vCPU = 1024)"
  type        = number
  default     = 1024
}

variable "ecs_memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 2048
}

variable "regllm_backend" {
  description = "App inference backend: groq | ollama | local"
  type        = string
  default     = "groq"
}

variable "cors_origins" {
  description = "CORS allowed origins (defaults to ALB DNS if empty)"
  type        = string
  default     = ""
}

variable "enable_https" {
  description = "Enable HTTPS on the ALB (requires acm_certificate_arn)"
  type        = bool
  default     = false
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS (required when enable_https = true)"
  type        = string
  default     = ""
}

variable "github_org" {
  description = "GitHub organisation or username that owns the repo (for OIDC trust)"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name (for OIDC trust), e.g. regllm"
  type        = string
  default     = "regllm"
}

variable "vllm_host_override" {
  description = "External vLLM/HF Inference Endpoint URL. When set, skips the EC2 inference module."
  type        = string
  default     = ""
}

variable "enable_inference" {
  description = "Launch the EC2 GPU inference instance (g4dn.xlarge + vLLM)"
  type        = bool
  default     = false
}

variable "base_model" {
  description = "HuggingFace model repo to serve (e.g. mistralai/Mistral-7B-Instruct-v0.3)"
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "lora_repo" {
  description = "HuggingFace LoRA adapter repo. Leave empty to serve the base model directly."
  type        = string
  default     = ""
}

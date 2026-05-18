variable "app_name"                    { type = string }
variable "environment"                 { type = string }
variable "aws_region"                  { type = string }
variable "ecs_cpu"                     { type = number }
variable "ecs_memory"                  { type = number }
variable "api_ecr_repository_url"      { type = string }
variable "frontend_ecr_repository_url" { type = string }
variable "private_subnet_ids"          { type = list(string) }
variable "app_security_group_id"       { type = string }
variable "api_target_group_arn"        { type = string }
variable "frontend_target_group_arn"   { type = string }
variable "rds_endpoint"                { type = string }
variable "db_password_secret_arn"      { type = string }
variable "groq_api_key_secret_arn"     { type = string }
variable "jwt_secret_arn"              { type = string }

variable "api_image_tag" {
  type    = string
  default = "latest"
}

variable "frontend_image_tag" {
  type    = string
  default = "latest"
}

variable "regllm_backend" {
  type    = string
  default = "groq"
}

variable "cors_origins" {
  type    = string
  default = ""
}

variable "hf_token_secret_arn" {
  description = "Secrets Manager ARN for the HuggingFace API token"
  type        = string
}

variable "vllm_host" {
  description = "Base URL of the vLLM inference server (e.g. http://10.0.11.x:8080). Empty string disables."
  type        = string
  default     = ""
}

variable "vllm_model" {
  description = "Model name to send in the OpenAI-compatible API request."
  type        = string
  default     = "kabesaml/regllm-qwen25-7b-banking-merged"
}

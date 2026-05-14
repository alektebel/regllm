variable "app_name"                 { type = string }
variable "environment"              { type = string }
variable "aws_region"               { type = string }
variable "ecs_cpu"                  { type = number }
variable "ecs_memory"               { type = number }
variable "ecr_repository_url"       { type = string }
variable "private_subnet_ids"       { type = list(string) }
variable "app_security_group_id"    { type = string }
variable "alb_target_group_arn"     { type = string }
variable "rds_endpoint"             { type = string }
variable "mlflow_bucket"            { type = string }
variable "weights_bucket"           { type = string }
variable "db_password_secret_arn"   { type = string }
variable "groq_api_key_secret_arn"  { type = string }
variable "efs_file_system_id"       { type = string }
variable "efs_access_point_id"      { type = string }
variable "regllm_backend"           { type = string; default = "groq" }
variable "mlflow_model_stage"       { type = string; default = "Production" }

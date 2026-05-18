variable "app_name"                    { type = string }
variable "environment"                 { type = string }
variable "aws_region"                  { type = string }
variable "subnet_id"                   { type = string }
variable "inference_security_group_id" { type = string }
variable "hf_token_secret_arn"         { type = string }

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

variable "root_volume_gb" {
  type    = number
  default = 100
}

variable "base_model" {
  type    = string
  default = "Qwen/Qwen2.5-7B-Instruct"
}

variable "lora_repo" {
  type    = string
  default = "kabesaml/regllm-qwen25-7b-banking-lora"
}

variable "private_ip" {
  description = "Fixed private IP for the inference instance (must be within the subnet CIDR). Stays stable across stop/start cycles."
  type        = string
  default     = "10.0.11.100"
}

variable "schedule_start_utc" {
  description = "Cron (UTC) to auto-start the instance. Empty string disables."
  type        = string
  default     = "cron(0 7 * * ? *)"  # 08:00 Madrid (UTC+1 winter)
}

variable "schedule_stop_utc" {
  description = "Cron (UTC) to auto-stop the instance. Empty string disables."
  type        = string
  default     = "cron(0 19 * * ? *)"  # 20:00 Madrid (UTC+1 winter)
}

variable "aws_region" {
  default = "eu-west-1"
}

variable "project" {
  default = "regllm-dqc"
}

variable "vpc_id" {
  description = "VPC ID for ECS deployment"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks and ALB"
  type        = list(string)
}

variable "cpu" {
  description = "Fargate task CPU units"
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory (MiB)"
  default     = 4096
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model ID for DQC generation"
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

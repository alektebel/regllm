output "repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.regllm.repository_url
}

output "repository_arn" {
  value = aws_ecr_repository.regllm.arn
}

output "api_repository_url" {
  description = "ECR repository URL for the API image"
  value       = aws_ecr_repository.api.repository_url
}

output "frontend_repository_url" {
  description = "ECR repository URL for the frontend image"
  value       = aws_ecr_repository.frontend.repository_url
}

output "api_repository_arn" {
  value = aws_ecr_repository.api.arn
}

output "frontend_repository_arn" {
  value = aws_ecr_repository.frontend.arn
}

output "app_url" {
  description = "Application URL (ALB DNS name)"
  value       = "http://${module.alb.dns_name}"
}

output "ecr_api_repository_url" {
  description = "ECR repository URL for the API image"
  value       = module.ecr.api_repository_url
}

output "ecr_frontend_repository_url" {
  description = "ECR repository URL for the frontend image"
  value       = module.ecr.frontend_repository_url
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = module.ecs.service_name
}

output "github_deploy_role_arn" {
  description = "IAM role ARN to set as AWS_DEPLOY_ROLE_ARN in GitHub secrets"
  value       = var.github_org != "" ? aws_iam_role.github_deploy[0].arn : "Set github_org variable to create the OIDC role"
}

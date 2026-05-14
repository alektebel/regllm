output "app_url" {
  description = "Application URL (ALB DNS name)"
  value       = "http://${module.alb.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = module.ecr.repository_url
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = module.s3.mlflow_bucket_name
}

output "model_weights_bucket" {
  description = "S3 bucket for raw model weights"
  value       = module.s3.weights_bucket_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

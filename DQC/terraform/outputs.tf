output "alb_dns" {
  value = aws_lb.main.dns_name
}

output "ecr_api_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_dqc_url" {
  value = aws_ecr_repository.dqc.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.main.name
}

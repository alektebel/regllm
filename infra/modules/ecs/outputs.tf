output "cluster_name" {
  value = aws_ecs_cluster.regllm.name
}

output "cluster_arn" {
  value = aws_ecs_cluster.regllm.arn
}

output "service_name" {
  value = aws_ecs_service.regllm_app.name
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.regllm_app.arn
}

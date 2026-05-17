output "cluster_name" {
  value = aws_ecs_cluster.regllm.name
}

output "cluster_arn" {
  value = aws_ecs_cluster.regllm.arn
}

output "service_name" {
  value = aws_ecs_service.regllm.name
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.regllm.arn
}

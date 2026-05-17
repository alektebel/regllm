output "dns_name" {
  value = aws_lb.regllm.dns_name
}

output "api_target_group_arn" {
  value = aws_lb_target_group.api.arn
}

output "frontend_target_group_arn" {
  value = aws_lb_target_group.frontend.arn
}

output "alb_arn" {
  value = aws_lb.regllm.arn
}

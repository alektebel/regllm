output "dns_name" {
  value = aws_lb.regllm.dns_name
}

output "target_group_arn" {
  value = aws_lb_target_group.app.arn
}

output "alb_arn" {
  value = aws_lb.regllm.arn
}

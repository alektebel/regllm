output "endpoint" {
  description = "RDS endpoint (host:port)"
  value       = aws_db_instance.regllm.endpoint
  sensitive   = true
}

output "address" {
  description = "RDS hostname only"
  value       = aws_db_instance.regllm.address
  sensitive   = true
}

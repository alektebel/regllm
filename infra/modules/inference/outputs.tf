output "instance_id" {
  value = aws_instance.inference.id
}

output "private_ip" {
  description = "Fixed private IP of the inference EC2 instance (stable across stop/start)"
  value       = var.private_ip
}

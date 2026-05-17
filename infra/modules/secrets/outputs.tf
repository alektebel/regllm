output "db_password_secret_arn" {
  value = aws_secretsmanager_secret.db_password.arn
}

output "groq_api_key_secret_arn" {
  value = aws_secretsmanager_secret.groq_api_key.arn
}

output "hf_token_secret_arn" {
  value = aws_secretsmanager_secret.hf_token.arn
}

output "jwt_secret_arn" {
  value = aws_secretsmanager_secret.jwt_secret.arn
}

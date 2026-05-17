# Recovery window: 30 days on prod, 0 (immediate delete) on dev
locals {
  recovery_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${var.app_name}/${var.environment}/db_password"
  recovery_window_in_days = local.recovery_days
}

resource "aws_secretsmanager_secret" "groq_api_key" {
  name                    = "${var.app_name}/${var.environment}/groq_api_key"
  recovery_window_in_days = local.recovery_days
}

resource "aws_secretsmanager_secret" "hf_token" {
  name                    = "${var.app_name}/${var.environment}/hf_token"
  recovery_window_in_days = local.recovery_days
}

resource "aws_secretsmanager_secret" "jwt_secret" {
  name                    = "${var.app_name}/${var.environment}/jwt_secret"
  recovery_window_in_days = local.recovery_days
}

# Note: secret *values* are populated manually (not in Terraform) to avoid
# storing credentials in state files or git history.

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = var.db_password_secret_arn
  secret_string = random_password.db.result
}

resource "aws_db_subnet_group" "regllm" {
  name       = "${var.app_name}-${var.environment}"
  subnet_ids = var.subnet_ids
}

resource "aws_db_parameter_group" "regllm" {
  name   = "${var.app_name}-pg16-${var.environment}"
  family = "postgres16"

  # Enable pg_vector extension
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_vector"
  }
}

resource "aws_db_instance" "regllm" {
  identifier             = "${var.app_name}-postgres-${var.environment}"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  max_allocated_storage  = 100
  db_name                = "regllm"
  username               = "regllm"
  password               = random_password.db.result
  parameter_group_name   = aws_db_parameter_group.regllm.name
  db_subnet_group_name   = aws_db_subnet_group.regllm.name
  vpc_security_group_ids = [var.db_security_group_id]
  skip_final_snapshot    = var.environment != "prod"
  deletion_protection    = var.environment == "prod"
  storage_encrypted      = true
  backup_retention_period = var.environment == "prod" ? 7 : 1
  multi_az               = var.environment == "prod"
}

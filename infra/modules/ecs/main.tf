data "aws_caller_identity" "current" {}

# ── CloudWatch log group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "regllm" {
  name              = "/ecs/${var.app_name}-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
}

# ── IAM: execution role (pull image, read secrets) ────────────────────────────
resource "aws_iam_role" "ecs_execution" {
  name = "${var.app_name}-ecs-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_basic" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [var.db_password_secret_arn, var.groq_api_key_secret_arn]
    }]
  })
}

# ── IAM: task role (S3, Secrets Manager, EFS) ─────────────────────────────────
resource "aws_iam_role" "ecs_task" {
  name = "${var.app_name}-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_permissions" {
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ModelArtifacts"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.mlflow_bucket}",
          "arn:aws:s3:::${var.mlflow_bucket}/*",
          "arn:aws:s3:::${var.weights_bucket}",
          "arn:aws:s3:::${var.weights_bucket}/*",
        ]
      },
      {
        Sid    = "SecretsRead"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [var.db_password_secret_arn, var.groq_api_key_secret_arn]
      },
      {
        Sid    = "EFSMount"
        Effect = "Allow"
        Action = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
          "elasticfilesystem:DescribeMountTargets",
        ]
        Resource = "*"
      },
    ]
  })
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "regllm" {
  name = "${var.app_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "regllm" {
  cluster_name       = aws_ecs_cluster.regllm.name
  capacity_providers = ["FARGATE"]
}

# ── Task Definition ───────────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "regllm_app" {
  family                   = "${var.app_name}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_cpu
  memory                   = var.ecs_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  # EFS volume for ChromaDB persistence
  volume {
    name = "chroma-data"
    efs_volume_configuration {
      file_system_id     = var.efs_file_system_id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = var.efs_access_point_id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([{
    name      = "${var.app_name}-app"
    image     = "${var.ecr_repository_url}:latest"
    essential = true

    portMappings = [{
      containerPort = 7860
      protocol      = "tcp"
    }]

    environment = [
      { name = "REGLLM_BACKEND",      value = var.regllm_backend },
      { name = "MLFLOW_MODEL_NAME",   value = "regllm-lora-adapter" },
      { name = "MLFLOW_MODEL_STAGE",  value = var.mlflow_model_stage },
      { name = "MLFLOW_ARTIFACT_ROOT", value = "s3://${var.mlflow_bucket}/mlflow" },
      { name = "POSTGRES_HOST",       value = split(":", var.rds_endpoint)[0] },
      { name = "POSTGRES_PORT",       value = "5432" },
      { name = "POSTGRES_DB",         value = "regllm" },
      { name = "POSTGRES_USER",       value = "regllm" },
      { name = "LOG_LEVEL",           value = "INFO" },
    ]

    secrets = [
      { name = "POSTGRES_PASSWORD", valueFrom = var.db_password_secret_arn },
      { name = "GROQ_API_KEY",      valueFrom = var.groq_api_key_secret_arn },
    ]

    mountPoints = [{
      sourceVolume  = "chroma-data"
      containerPath = "/app/vector_db/chroma_db"
      readOnly      = false
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.regllm.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "app"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:7860 || exit 1"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 180
    }
  }])
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "regllm_app" {
  name            = "${var.app_name}-app-${var.environment}"
  cluster         = aws_ecs_cluster.regllm.id
  task_definition = aws_ecs_task_definition.regllm_app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.app_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.alb_target_group_arn
    container_name   = "${var.app_name}-app"
    container_port   = 7860
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  # Allow ECS to manage the task definition version
  lifecycle {
    ignore_changes = [task_definition]
  }
}

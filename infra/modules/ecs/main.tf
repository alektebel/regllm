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
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = [
        var.db_password_secret_arn,
        var.groq_api_key_secret_arn,
        var.jwt_secret_arn,
        var.hf_token_secret_arn,
      ]
    }]
  })
}

# ── IAM: task role (Secrets Manager runtime access) ───────────────────────────
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
    Statement = [{
      Sid    = "SecretsRead"
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = [
        var.db_password_secret_arn,
        var.groq_api_key_secret_arn,
        var.jwt_secret_arn,
        var.hf_token_secret_arn,
      ]
    }]
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

# ── Task Definition (api + frontend sidecar) ──────────────────────────────────
locals {
  alb_dns = var.cors_origins != "" ? var.cors_origins : "http://localhost:3000"
}

resource "aws_ecs_task_definition" "regllm" {
  family                   = "${var.app_name}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_cpu
  memory                   = var.ecs_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    # ── FastAPI (api) ──────────────────────────────────────────────────────
    {
      name      = "${var.app_name}-api"
      image     = "${var.api_ecr_repository_url}:${var.api_image_tag}"
      essential = true

      portMappings = [{
        containerPort = 8000
        protocol      = "tcp"
      }]

      environment = [
        { name = "REGLLM_BACKEND", value = var.regllm_backend },
        { name = "VLLM_HOST",      value = var.vllm_host },
        { name = "POSTGRES_HOST",  value = split(":", var.rds_endpoint)[0] },
        { name = "POSTGRES_PORT",  value = "5432" },
        { name = "POSTGRES_DB",    value = "regllm" },
        { name = "POSTGRES_USER",  value = "regllm" },
        { name = "CORS_ORIGINS",   value = local.alb_dns },
        { name = "LOG_LEVEL",      value = "INFO" },
        { name = "JWT_EXPIRE_HOURS", value = "168" },
      ]

      secrets = [
        { name = "POSTGRES_PASSWORD", valueFrom = var.db_password_secret_arn },
        { name = "GROQ_API_KEY",      valueFrom = var.groq_api_key_secret_arn },
        { name = "JWT_SECRET",        valueFrom = var.jwt_secret_arn },
        { name = "HF_TOKEN",          valueFrom = var.hf_token_secret_arn },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.regllm.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    },

    # ── Next.js (frontend) ─────────────────────────────────────────────────
    {
      name      = "${var.app_name}-frontend"
      image     = "${var.frontend_ecr_repository_url}:${var.frontend_image_tag}"
      essential = true

      portMappings = [{
        containerPort = 3000
        protocol      = "tcp"
      }]

      environment = [
        # Browser-side: points to the ALB (same host, different path)
        { name = "NEXT_PUBLIC_API_URL", value = local.alb_dns },
        # Server-side SSR: reaches the API container via localhost (sidecar)
        { name = "INTERNAL_API_URL",    value = "http://localhost:8000" },
      ]

      dependsOn = [{
        containerName = "${var.app_name}-api"
        condition     = "HEALTHY"
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.regllm.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "frontend"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3000 || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "regllm" {
  name            = "${var.app_name}-app-${var.environment}"
  cluster         = aws_ecs_cluster.regllm.id
  task_definition = aws_ecs_task_definition.regllm.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.app_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.api_target_group_arn
    container_name   = "${var.app_name}-api"
    container_port   = 8000
  }

  load_balancer {
    target_group_arn = var.frontend_target_group_arn
    container_name   = "${var.app_name}-frontend"
    container_port   = 3000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  # ECS manages the task definition revision after the first deploy
  lifecycle {
    ignore_changes = [task_definition]
  }
}

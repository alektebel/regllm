# ── AMI: Deep Learning Base (Ubuntu 22.04) — NVIDIA drivers + Docker pre-installed
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ── IAM role so the instance can read Secrets Manager ─────────────────────────
resource "aws_iam_role" "inference" {
  name = "${var.app_name}-inference-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "inference_secrets" {
  name = "secrets-access"
  role = aws_iam_role.inference.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "secretsmanager:GetSecretValue"
      Resource = [var.hf_token_secret_arn]
    }]
  })
}

resource "aws_iam_instance_profile" "inference" {
  name = "${var.app_name}-inference-${var.environment}"
  role = aws_iam_role.inference.name
}

# ── EC2 instance ───────────────────────────────────────────────────────────────
resource "aws_instance" "inference" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  private_ip             = var.private_ip
  vpc_security_group_ids = [var.inference_security_group_id]
  iam_instance_profile   = aws_iam_instance_profile.inference.name

  root_block_device {
    volume_size = var.root_volume_gb  # model weights + Docker images cached here
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data.sh", {
    hf_token_secret_arn = var.hf_token_secret_arn
    aws_region          = var.aws_region
    base_model          = var.base_model
    lora_repo           = var.lora_repo
  })

  tags = {
    Name        = "${var.app_name}-inference-${var.environment}"
    Environment = var.environment
  }
}

# ── Auto stop/start scheduler ──────────────────────────────────────────────────

resource "aws_iam_role" "scheduler" {
  name = "${var.app_name}-inference-scheduler-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "scheduler.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "scheduler_ec2" {
  name = "ec2-start-stop"
  role = aws_iam_role.scheduler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ec2:StartInstances", "ec2:StopInstances"]
      Resource = aws_instance.inference.arn
    }]
  })
}

resource "aws_scheduler_schedule" "stop" {
  name       = "${var.app_name}-inference-stop-${var.environment}"
  group_name = "default"

  flexible_time_window { mode = "OFF" }

  schedule_expression          = var.schedule_stop_utc
  schedule_expression_timezone = "Europe/Madrid"

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ec2:stopInstances"
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({ InstanceIds = [aws_instance.inference.id] })
  }
}

resource "aws_scheduler_schedule" "start" {
  name       = "${var.app_name}-inference-start-${var.environment}"
  group_name = "default"

  flexible_time_window { mode = "OFF" }

  schedule_expression          = var.schedule_start_utc
  schedule_expression_timezone = "Europe/Madrid"

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:ec2:startInstances"
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({ InstanceIds = [aws_instance.inference.id] })
  }
}

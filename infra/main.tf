# ── VPC (simple 2-AZ setup) ───────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

locals {
  azs          = slice(data.aws_availability_zones.available.names, 0, 2)
  public_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
  private_cidrs = ["10.0.11.0/24", "10.0.12.0/24"]
}

data "aws_availability_zones" "available" { state = "available" }

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_cidrs[count.index]
  availability_zone = local.azs[count.index]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# NAT Gateway for private subnet egress (pip installs, model downloads)
resource "aws_eip" "nat" { domain = "vpc" }

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ── Security Groups ───────────────────────────────────────────────────────────

resource "aws_security_group" "alb" {
  name   = "${var.app_name}-alb-${var.environment}"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name   = "${var.app_name}-app-${var.environment}"
  vpc_id = aws_vpc.main.id

  # FastAPI
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  # Next.js
  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "db" {
  name   = "${var.app_name}-db-${var.environment}"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── Modules ───────────────────────────────────────────────────────────────────

module "ecr" {
  source      = "./modules/ecr"
  app_name    = var.app_name
  environment = var.environment
}

module "s3" {
  source      = "./modules/s3"
  app_name    = var.app_name
  environment = var.environment
}

module "secrets" {
  source      = "./modules/secrets"
  app_name    = var.app_name
  environment = var.environment
}

module "rds" {
  source                 = "./modules/rds"
  app_name               = var.app_name
  environment            = var.environment
  db_instance_class      = var.db_instance_class
  db_security_group_id   = aws_security_group.db.id
  subnet_ids             = aws_subnet.private[*].id
  db_password_secret_arn = module.secrets.db_password_secret_arn
}

module "alb" {
  source                = "./modules/alb"
  app_name              = var.app_name
  environment           = var.environment
  vpc_id                = aws_vpc.main.id
  public_subnet_ids     = aws_subnet.public[*].id
  alb_security_group_id = aws_security_group.alb.id
  enable_https          = var.enable_https
  acm_certificate_arn   = var.acm_certificate_arn
}

module "ecs" {
  source                       = "./modules/ecs"
  app_name                     = var.app_name
  environment                  = var.environment
  aws_region                   = var.aws_region
  ecs_cpu                      = var.ecs_cpu
  ecs_memory                   = var.ecs_memory
  api_ecr_repository_url       = module.ecr.api_repository_url
  frontend_ecr_repository_url  = module.ecr.frontend_repository_url
  private_subnet_ids           = aws_subnet.private[*].id
  app_security_group_id        = aws_security_group.app.id
  api_target_group_arn         = module.alb.api_target_group_arn
  frontend_target_group_arn    = module.alb.frontend_target_group_arn
  rds_endpoint                 = module.rds.endpoint
  db_password_secret_arn       = module.secrets.db_password_secret_arn
  groq_api_key_secret_arn      = module.secrets.groq_api_key_secret_arn
  jwt_secret_arn               = module.secrets.jwt_secret_arn
  regllm_backend               = var.regllm_backend
  cors_origins                 = var.cors_origins
}

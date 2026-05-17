# ── GitHub Actions OIDC IAM Role ──────────────────────────────────────────────
#
# The GitHub OIDC provider already exists in this account (created manually).
# We reference it via a data source and create the deploy role that GitHub
# Actions will assume.
#
# Set var.github_org in your .tfvars file to activate this block.
# After apply, copy the output `github_deploy_role_arn` value into the
# GitHub secret AWS_DEPLOY_ROLE_ARN.
#
# Also set the GitHub secret AWS_REGION = eu-west-1

data "aws_caller_identity" "oidc" {}

# Look up the existing OIDC provider (created outside Terraform)
data "aws_iam_openid_connect_provider" "github" {
  count = var.github_org != "" ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_role" "github_deploy" {
  count = var.github_org != "" ? 1 : 0

  name = "${var.app_name}-github-deploy-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = data.aws_iam_openid_connect_provider.github[0].arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = [
            "repo:${var.github_org}/${var.github_repo}:ref:refs/heads/main",
            "repo:${var.github_org}/${var.github_repo}:ref:refs/heads/*",
          ]
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_deploy_ecr" {
  count = var.github_org != "" ? 1 : 0
  role  = aws_iam_role.github_deploy[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
        ]
        Resource = [
          module.ecr.api_repository_arn,
          module.ecr.frontend_repository_arn,
        ]
      },
      {
        Sid    = "ECSDescribe"
        Effect = "Allow"
        Action = [
          "ecs:DescribeTaskDefinition",
          "ecs:DescribeServices",
          "ecs:ListTasks",
        ]
        Resource = "*"
      },
      {
        Sid    = "ECSDeploy"
        Effect = "Allow"
        Action = [
          "ecs:RegisterTaskDefinition",
          "ecs:UpdateService",
        ]
        Resource = "*"
      },
      {
        Sid    = "PassECSIAMRoles"
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          "arn:aws:iam::${data.aws_caller_identity.oidc.account_id}:role/${var.app_name}-ecs-execution-${var.environment}",
          "arn:aws:iam::${data.aws_caller_identity.oidc.account_id}:role/${var.app_name}-ecs-task-${var.environment}",
        ]
      },
    ]
  })
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.app_name}-mlflow-artifacts-${var.environment}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket                  = aws_s3_bucket.mlflow_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "model_weights" {
  bucket = "${var.app_name}-model-weights-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_weights" {
  bucket = aws_s3_bucket.model_weights.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "model_weights" {
  bucket                  = aws_s3_bucket.model_weights.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

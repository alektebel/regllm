output "mlflow_bucket_name" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}

output "mlflow_bucket_arn" {
  value = aws_s3_bucket.mlflow_artifacts.arn
}

output "weights_bucket_name" {
  value = aws_s3_bucket.model_weights.bucket
}

output "weights_bucket_arn" {
  value = aws_s3_bucket.model_weights.arn
}

# Remote state in S3.
# Bootstrap once before first `terraform init`:
#   aws s3 mb s3://regllm-terraform-state --region eu-west-1
#   aws dynamodb create-table \
#     --table-name regllm-terraform-locks \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST \
#     --region eu-west-1

terraform {
  backend "s3" {
    bucket         = "regllm-terraform-state"
    key            = "regllm/terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "regllm-terraform-locks"
    encrypt        = true
  }
}

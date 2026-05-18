environment       = "dev"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.micro"
ecs_cpu           = 1024
ecs_memory        = 2048
regllm_backend      = "vllm"
enable_https        = false
cors_origins        = "http://regllm-dev-725928702.eu-west-1.elb.amazonaws.com"
enable_inference    = false
vllm_host_override  = "https://k38shd29dpu9sya5.us-east-1.aws.endpoints.huggingface.cloud"

# Set your GitHub org/username to create the OIDC deploy role
# github_org  = "alektebel"
# github_repo = "regllm"

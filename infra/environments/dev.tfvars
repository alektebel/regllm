environment       = "dev"
aws_region        = "eu-west-1"
db_instance_class = "db.t3.micro"
ecs_cpu           = 1024
ecs_memory        = 2048
regllm_backend    = "groq"
enable_https      = false
cors_origins      = "http://regllm-dev-725928702.eu-west-1.elb.amazonaws.com"

# Set your GitHub org/username to create the OIDC deploy role
# github_org  = "alektebel"
# github_repo = "regllm"

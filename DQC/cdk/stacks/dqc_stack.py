"""RegLLM DQC — AWS CDK stack.

Deploys the DQC application on ECS Fargate behind an ALB, with a Gemini
(or Bedrock) LLM backend for the API container.  Self-contained: creates
its own VPC, ECR repos, IAM roles, ALB, ECS cluster and service.

Optional context (pass via ``cdk deploy -c key=value``):
  project          — resource name prefix   (default: regllm-dqc)
  aws_region       — AWS region             (default: eu-west-1)
  cpu              — Fargate CPU units       (default: 1024)
  memory           — Fargate memory MiB      (default: 4096)
  bedrock_model_id — Bedrock model           (default: eu.amazon.nova-micro-v1:0)
  gemini_api_key   — Google Gemini API key; when set, switches LLM backend to gemini
  gemini_model     — Gemini model            (default: gemini-2.5-pro)

Use ``DQC/cdk/deploy.sh`` for a one-command deploy that also builds and
pushes the Docker images to the CDK-created ECR repos.
"""

from __future__ import annotations

from aws_cdk import (
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
    CfnOutput,
)
from constructs import Construct


class DqcStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Context / parameters ───────────────────────────────────────────
        project = self.node.try_get_context("project") or "regllm-dqc"
        bedrock_model_id = (
            self.node.try_get_context("bedrock_model_id")
            or "eu.amazon.nova-micro-v1:0"
        )
        gemini_api_key = self.node.try_get_context("gemini_api_key") or ""
        gemini_model = self.node.try_get_context("gemini_model") or "gemini-2.5-pro"
        cpu = int(self.node.try_get_context("cpu") or 1024)
        memory = int(self.node.try_get_context("memory") or 4096)

        # ── VPC — create a dedicated public-only VPC (no NAT gateways).
        # The Fargate tasks get public IPs (assign_public_ip=True), so they
        # can pull from ECR / reach the Gemini API directly without NAT.
        vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            ],
            nat_gateways=0,
        )
        subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)

        # ── ECR ────────────────────────────────────────────────────────────
        ecr_api = ecr.Repository(
            self, "EcrApi",
            repository_name=f"{project}-api",
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
        )
        ecr_dqc = ecr.Repository(
            self, "EcrDqc",
            repository_name=f"{project}-dqc",
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
        )

        # ── CloudWatch ─────────────────────────────────────────────────────
        log_group = logs.LogGroup(
            self, "Logs",
            log_group_name=f"/ecs/{project}",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── IAM ────────────────────────────────────────────────────────────
        exec_role = iam.Role(
            self, "EcsExecRole",
            role_name=f"{project}-ecs-exec",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                ),
            ],
        )

        task_role = iam.Role(
            self, "EcsTaskRole",
            role_name=f"{project}-ecs-task",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        # Cross-region inference profiles (e.g. eu.amazon.nova-micro-v1:0) route
        # calls across multiple regions.  The policy must cover both the inference
        # profile ARN (account-scoped, home region) and the foundation-model ARNs
        # in every region the profile may route to.
        task_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            resources=[
                # Inference profile (account-scoped)
                f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{bedrock_model_id}",
                # Foundation models reachable by the profile (all regions wildcard)
                f"arn:aws:bedrock:*::foundation-model/*",
            ],
        ))

        # ── Security Groups ────────────────────────────────────────────────
        alb_sg = ec2.SecurityGroup(self, "AlbSg", vpc=vpc,
                                   security_group_name=f"{project}-alb",
                                   allow_all_outbound=True)
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80))

        task_sg = ec2.SecurityGroup(self, "TaskSg", vpc=vpc,
                                    security_group_name=f"{project}-task",
                                    allow_all_outbound=True)
        task_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(80))
        task_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8000))

        # ── ALB ────────────────────────────────────────────────────────────
        alb = elbv2.ApplicationLoadBalancer(
            self, "Alb",
            load_balancer_name=project,
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=subnets,
        )

        # ── ECS Cluster ────────────────────────────────────────────────────
        cluster = ecs.Cluster(self, "Cluster", cluster_name=project, vpc=vpc)

        # ── Task Definition ────────────────────────────────────────────────
        task_def = ecs.FargateTaskDefinition(
            self, "TaskDef",
            family=project,
            cpu=cpu,
            memory_limit_mib=memory,
            execution_role=exec_role,
            task_role=task_role,
        )

        # API container
        api_env: dict[str, str] = {
            "CORS_ORIGINS": "*",
            "BEDROCK_MODEL_ID": bedrock_model_id,
            "BEDROCK_REGION": self.region,
        }
        if gemini_api_key:
            api_env["REGLLM_LLM"] = "gemini"
            api_env["GEMINI_API_KEY"] = gemini_api_key
            api_env["GEMINI_MODEL"] = gemini_model
        else:
            api_env["REGLLM_LLM"] = "bedrock"

        api_container = task_def.add_container(
            "api",
            image=ecs.ContainerImage.from_ecr_repository(ecr_api, "latest"),
            essential=True,
            environment=api_env,
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="api",
                log_group=log_group,
            ),
        )
        api_container.add_port_mappings(ecs.PortMapping(container_port=8000))

        # DQC frontend container — in Fargate, sidecars share localhost
        dqc_container = task_def.add_container(
            "dqc",
            image=ecs.ContainerImage.from_ecr_repository(ecr_dqc, "latest"),
            essential=True,
            environment={
                "API_UPSTREAM": "localhost:8000",
            },
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="dqc",
                log_group=log_group,
            ),
        )
        dqc_container.add_port_mappings(ecs.PortMapping(container_port=80))

        # ── ECS Service ────────────────────────────────────────────────────
        service = ecs.FargateService(
            self, "Service",
            service_name=project,
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[task_sg],
            vpc_subnets=subnets,
        )

        # ── ALB Target Group + Listener ────────────────────────────────────
        listener = alb.add_listener("Http", port=80, open=False)
        listener.add_targets(
            "DqcTarget",
            port=80,
            targets=[service.load_balancer_target(
                container_name="dqc",
                container_port=80,
            )],
            health_check=elbv2.HealthCheck(path="/"),
        )

        # ── Outputs ────────────────────────────────────────────────────────
        CfnOutput(self, "AlbDns", value=alb.load_balancer_dns_name)
        CfnOutput(self, "EcrApiUrl", value=ecr_api.repository_uri)
        CfnOutput(self, "EcrDqcUrl", value=ecr_dqc.repository_uri)
        CfnOutput(self, "EcsCluster", value=cluster.cluster_name)
        CfnOutput(self, "EcsService", value=service.service_name)

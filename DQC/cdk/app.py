#!/usr/bin/env python3
"""CDK app entry point for RegLLM DQC.

The stack deploys into an EXISTING VPC (`-c vpc_id=...` required;
`DQC/cdk/deploy.sh` resolves the account's default VPC automatically).
Context overrides:

    -c vpc_id=vpc-...            # required: VPC to deploy into
    -c subnet_ids=subnet-a,...   # subnets for ALB + tasks
    -c aws_account=123456789012  # required for the VPC lookup
    -c project=regllm-dqc
    -c aws_region=eu-west-1
    -c gemini_api_key=...        # switches LLM backend to gemini
    -c gemini_model=gemini-2.5-pro
    -c bedrock_model_id=eu.amazon.nova-micro-v1:0
    -c cpu=1024
    -c memory=4096
"""

import aws_cdk as cdk

from stacks.dqc_stack import DqcStack

app = cdk.App()

DqcStack(
    app,
    "RegllmDqcStack",
    env=cdk.Environment(
        account=app.node.try_get_context("aws_account") or cdk.Aws.ACCOUNT_ID,
        region=app.node.try_get_context("aws_region") or "eu-west-1",
    ),
)

app.synth()

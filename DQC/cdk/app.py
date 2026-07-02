#!/usr/bin/env python3
"""CDK app entry point for RegLLM DQC.

Usage:
    cdk deploy -c vpc_id=vpc-xxx -c subnet_ids=subnet-a,subnet-b

Optional context overrides:
    -c project=regllm-dqc
    -c bedrock_model_id=anthropic.claude-3-haiku-20240307-v1:0
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
        region=app.node.try_get_context("aws_region") or "eu-west-1",
    ),
)

app.synth()

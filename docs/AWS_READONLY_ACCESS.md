# AWS read-only access — see the deployed PoC

Permissions to give an IAM **user** so they can *view* the deployed DQC
resources (stack, ECS service, logs, app URL) without being able to change
or deploy anything.

> Note: to **use** the app nobody needs AWS permissions — the ALB URL is
> public HTTP. IAM only governs visibility of the AWS-side resources.

## Simplest

Attach the AWS-managed **`ReadOnlyAccess`** policy to the user. One click,
covers everything below.

## Scoped to just this app

| To see… | Permissions |
|---|---|
| The app URL (ALB DNS) | `elasticloadbalancing:Describe*` |
| The running service / tasks | `ecs:List*`, `ecs:Describe*` |
| The CloudFormation stack | `cloudformation:Describe*`, `cloudformation:List*`, `cloudformation:GetTemplate` |
| The container images | `ecr:Describe*`, `ecr:List*` |
| Logs (`/ecs/regllm-dqc`) | `logs:Get*`, `logs:Describe*`, `logs:FilterLogEvents`, `logs:StartQuery`, `logs:GetQueryResults` |
| VPC / subnets / security groups | `ec2:Describe*` |
| Which Bedrock model is enabled | `bedrock:ListFoundationModels`, `bedrock:GetFoundationModelAvailability` |

Ready-to-paste inline policy (same as
[`DQC/cdk/iam-readonly-policy.json`](../DQC/cdk/iam-readonly-policy.json)):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "SeeDqcPoc",
    "Effect": "Allow",
    "Action": [
      "cloudformation:Describe*", "cloudformation:List*", "cloudformation:GetTemplate",
      "ecs:List*", "ecs:Describe*",
      "ecr:Describe*", "ecr:List*",
      "elasticloadbalancing:Describe*",
      "ec2:Describe*",
      "logs:Get*", "logs:Describe*", "logs:FilterLogEvents", "logs:StartQuery", "logs:GetQueryResults",
      "bedrock:ListFoundationModels", "bedrock:GetFoundationModelAvailability"
    ],
    "Resource": "*"
  }]
}
```

## Notes

- All actions are `List`/`Describe`/`Get` — non-mutating. The user can view
  stack, service health, logs, and the app URL but cannot change or deploy
  anything.
- `Resource: "*"` is used because describe/list actions largely don't
  support resource-level scoping; the actions themselves are read-only, so
  this stays safe.
- This is **read-only**; deploying the PoC needs the broader deployer
  permissions in [`AWS_POC_SETUP.md`](AWS_POC_SETUP.md#0b-permissions-you-need).

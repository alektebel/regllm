#!/bin/bash
set -e
exec > /var/log/user-data.log 2>&1
echo "=== vLLM inference server startup $(date) ==="

# Fetch HF token from Secrets Manager (empty string if not set)
HF_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id "${hf_token_secret_arn}" \
  --query SecretString \
  --output text \
  --region "${aws_region}" 2>/dev/null || echo "")

# Create model cache dir (persists on EBS — model only downloads once)
mkdir -p /data/hf-cache
chmod 777 /data/hf-cache

# Write env file for the service (no secrets in systemd unit)
cat > /etc/vllm.env << EOF
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
HF_TOKEN=$HF_TOKEN
EOF
chmod 600 /etc/vllm.env

# vLLM systemd service — runs the official Docker image
cat > /etc/systemd/system/vllm.service << 'UNIT'
[Unit]
Description=vLLM OpenAI-compatible inference server
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=15
ExecStartPre=-/usr/bin/docker stop vllm
ExecStartPre=-/usr/bin/docker rm vllm
%{ if lora_repo != "" ~}
ExecStart=/usr/bin/docker run \
  --name vllm \
  --gpus all \
  --ipc=host \
  --env-file /etc/vllm.env \
  -v /data/hf-cache:/root/.cache/huggingface \
  -p 8080:8000 \
  vllm/vllm-openai:latest \
  --model ${base_model} \
  --enable-lora \
  --lora-modules regllm=${lora_repo} \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 \
  --dtype half \
  --enforce-eager
%{ else ~}
ExecStart=/usr/bin/docker run \
  --name vllm \
  --gpus all \
  --ipc=host \
  --env-file /etc/vllm.env \
  -v /data/hf-cache:/root/.cache/huggingface \
  -p 8080:8000 \
  vllm/vllm-openai:latest \
  --model ${base_model} \
  --served-model-name regllm \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 \
  --dtype half \
  --enforce-eager
%{ endif ~}

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable vllm
systemctl start vllm

echo "=== vLLM service started — model download may take 20-30 min on first boot ==="

-- Create the MLflow metadata database alongside the app database.
-- This runs automatically on first postgres container startup.
SELECT 'CREATE DATABASE regllm_mlflow'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'regllm_mlflow'
)\gexec

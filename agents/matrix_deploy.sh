# Start cluster
matrix start_cluster --enable_grafana --add_workers 8 --slurm "{'slurm_account': 'sage', 'slurm_qos': 'sage_high'}" --force_new_head
matrix start_cluster --add_workers 8 --slurm "{'slurm_account': 'sage', 'slurm_qos': 'sage_high'}"


# Llama 3 models
matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.2-3B-Instruct', 'min_replica': 1, 'name': '3B'}]"
matrix check_health --app_name 3B

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'min_replica': 1, 'name': '8B'}]"
matrix check_health --app_name 8B

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-70B-Instruct', 'min_replica': 4, 'name': '70B'}]"
matrix check_health --app_name 70B

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.3-70B-Instruct', 'min_replica': 1, 'name': '3_3_70B'}]"
matrix check_health --app_name 3_3_70B

matrix deploy_applications --applications '[{"name": "70B_grpc", "model_name": "/datasets/pretrained-llms/Llama-3.3-70B-Instruct", "min_replica": 1, "model_size": "70B", "use_grpc": "true"}]'

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-405B-Instruct', 'min_replica': 2, 'name': '405B'}]"
matrix check_health --app_name 405B

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-3.1-405B-Instruct-FP8', 'min_replica': 2, 'name': '405B-FP8'}]"
matrix check_health --app_name 405B-FP8


# Llama 4 models
HF_HUB_OFFLINE=1 matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'min_replica': 2, 'name': 'scout'}]"
matrix check_health --app_name scout

matrix deploy_applications --action add --applications "[{'model_name': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', 'min_replica': 1, 'name': 'maverick'}]"
matrix check_health --app_name maverick


# Deepseek R1
matrix deploy_applications --action add --applications "[{'model_name': 'deepseek-ai/DeepSeek-R1', 'pipeline-parallel-size': 2, 'app_type': sglang_llm, 'name': 'r1'}]"


# Qwen 2.5 models
matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen2.5-7B-Instruct', 'min_replica': 2, 'name': 'Qwen2.5-7B'}]"
matrix check_health --app_name Qwen2.5-7B

matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen2.5-32B-Instruct', 'min_replica': 2, 'name': 'Qwen2.5-32B'}]"
matrix check_health --app_name Qwen2.5-32B

matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen2.5-72B-Instruct', 'min_replica': 2, 'name': 'Qwen2.5-72B'}]"
matrix check_health --app_name Qwen2.5-72B


# Qwen 3 models
matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen3-8B-Instruct', 'min_replica': 2, 'name': 'Qwen3-8B'}]"
matrix check_health --app_name Qwen3-8B

matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen3-32B-Instruct', 'min_replica': 2, 'name': 'Qwen3-32B'}]"
matrix check_health --app_name Qwen3-32B

matrix deploy_applications --action add --applications "[{'model_name': 'Qwen/Qwen3-235B-A22B', 'min_replica': 2, 'name': 'Qwen3-235B-A22B'}]"
matrix check_health --app_name Qwen3-235B-A22B


# Remove all applications
matrix deploy_applications --applications ''

# Check status of all applications
matrix status

matrix stop_cluster


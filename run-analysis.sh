#! /usr/bin/env sh

export S3_BUCKET="bedrock-data-source-tomjhnsn"
export SERVICE_ROLE="arn:aws:iam::438465170454:role/service-role/batch-tasks"
export CONCURRENT_TASK_LIMIT=10
export BASE_DIR="ai-rag-repos/js"

poetry run analyse-code --work_dir=./work_dir/js-1 \
    --path=/Users/tomjhnsn/workplace/ai-rag-repos/js \
    --batch --llm=sonnetV2 --build --no-query --client=js

# poetry run analyse-code --work_dir=./work_dir/rust-1 \
#     --path=/Users/tomjhnsn/workplace/ai-rag-repos/rust \
#     --batch --llm=sonnetV2 --build --no-query --client=rust

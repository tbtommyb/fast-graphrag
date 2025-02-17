#! /usr/bin/env sh

export S3_BUCKET="bedrock-data-source-tomjhnsn"
export SERVICE_ROLE="arn:aws:iam::438465170454:role/service-role/batch-tasks"
export CONCURRENT_TASK_LIMIT=10
export BASE_DIR="src/AVLivingRoomClient"

poetry run analyse-code --work_dir=./batch_prompts/avlrc-details-filemaps \
    --path=/Users/tomjhnsn/workplace/avlrc-dev/src/AVLivingRoomClient/packages/details \
    --batch --llm=sonnet --build --query

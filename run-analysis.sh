#!/bin/bash

# Store the original directory
ORIGINAL_DIR=$(pwd)

# Check if argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [js|rust]"
    exit 1
fi

# Validate argument
if [ "$1" != "js" ] && [ "$1" != "rust" ]; then
    echo "Error: Argument must be either 'js' or 'rust'"
    exit 1
fi

# Set variables based on argument
export S3_BUCKET="bedrock-data-source-ai-rag-bucket-us-west-2"
export SERVICE_ROLE="arn:aws:iam::418272767634:role/ChatLRC-EC2"
export CONCURRENT_TASK_LIMIT=10
export BASE_DIR="ai-rag-repos/$1"
export PYENV_VERSION=3.11.11
export PYENV_ROOT=/home/ubuntu/.pyenv

# Create analysis worktrees
SOURCE_DIR="/data/workplace/ai-rag-repos/$1"
ANALYSIS_BASE_DIR="/data/workplace/analysis-worktree-$1"
CURRENT_DATE=$(date +%Y-%m-%d)

# Create base analysis directory if it doesn't exist
mkdir -p "$ANALYSIS_BASE_DIR"

# Only create worktrees if analysis directory is empty
if [ -z "$(ls -A $ANALYSIS_BASE_DIR)" ]; then
    echo "Creating fresh worktrees..."
    # Process each repository
    for repo in "$SOURCE_DIR"/*; do
        if [ -d "$repo" ] && [ -d "$repo/.git" ]; then
            repo_name=$(basename "$repo")
            analysis_path="$ANALYSIS_BASE_DIR/$repo_name"

            echo "Processing repository: $repo_name"

            # Create new worktree
            echo "Creating new worktree for $repo_name..."
            cd "$repo"
            git worktree add "$analysis_path" HEAD
        elif [ -L "$repo" ]; then
            # Handle symlinks
            echo "Creating symlink for $(basename "$repo")"
            ln -sf "$repo" "$ANALYSIS_BASE_DIR/$(basename "$repo")"
        fi
    done
else
    echo "Using existing worktrees in $ANALYSIS_BASE_DIR"
fi

# Return to original directory before running analysis
cd "$ORIGINAL_DIR"

# Run analysis using the worktree path instead of original repo
pyenv shell 3.11.11
poetry run analyse-code --work_dir="./work_dir/${CURRENT_DATE}-$1" \
        --path="$ANALYSIS_BASE_DIR" \
        --batch --llm=sonnetV2 --build --no-query --client="$1"

# Only clean up if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully. Cleaning up worktrees..."
    for repo in "$SOURCE_DIR"/*; do
        if [ -d "$repo" ] && [ -d "$repo/.git" ]; then
            repo_name=$(basename "$repo")
            analysis_path="$ANALYSIS_BASE_DIR/$repo_name"

            cd "$repo"
            git worktree remove -f "$analysis_path"
        fi
    done

    # Return to original directory and remove the base analysis directory
    cd "$ORIGINAL_DIR"
    rm -rf "$ANALYSIS_BASE_DIR"
else
    echo "Analysis did not complete successfully. Leaving worktrees in place for restart."
    cd "$ORIGINAL_DIR"
fi

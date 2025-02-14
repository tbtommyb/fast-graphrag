## AVLRCFastRag

### Setup

```
$ poetry install
```

If you have issues with `libhnsw` not building, try creating a Python virtual env using Python 3.11 and ensure that `gcc` is installed in the venv.

### Usage

Look in `run.sh` to see an example of how to run.

Currently `analyse-code.py` is written assuming that you have an `ada` profile called `PROFILE_NAME` that has IAM permissions to access Bedrock.

Note that in `analyse-code.py` we need to provide API endponts to access embedding and chat LLMs. You will need to have enabled access to these LLMs in your account's Bedrock profile.

You can use the package [`bedrock-access-gateway`](https://github.com/aws-samples/bedrock-access-gateway) to proxy requests from an OpenAI-compatible API endpoint to the Bedrock CDK. (Note: I have a patched version that includes support for Anthropic's tool use so that the Anthropic LLM client can generate structured output. I will upload it to a separate repo and link).

```
export S3_BUCKET="bedrock-data-source-tomjhnsn" # the S3 bucket that batch prompts will be written to
export SERVICE_ROLE="arn:aws:iam::438465170454:role/service-role/batch-tasks" # the service role to access S3
export CONCURRENT_TASK_LIMIT=10 # might need lower if your account is not production

poetry run analyse-code --work_dir=./knowledge-graphs/avlrc-no-glean-ts-only-batched \
    --path=/Users/tomjhnsn/workplace/avlrc-dev/src/AVLivingRoomClient/packages \
    --batch --llm=sonnet --no-build --query
```
`fast-graphrag` will output its processed data to `work_dir` so if you call it with the same `work_dir` it should reuse previously computed data.

`path` is the source path of the data you want to analyse. Currently `[ts, tsx]` extensions are hardcoded in `analyse-code.py` but this can be easily changed to suit your codebase. Be careful not to accidentally include non-code files like auto-generated data.

`batch/no-batch` controls whether to run the build in Bedrock batch mode (S3 access required) or sequentially. Batch is much faster.

`build/no-build` controls whether it tries to build the knowledge graph or just load a previously-computed one.

`query/no-query` controls whether it enters an interactive query mode after loading the knowledge graph.

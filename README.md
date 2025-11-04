# Neuro-SAN Benchmarking

A benchmarking suite for evaluating multi-agent reasoning systems on arithmetic problems using Neuro-SAN.

## Overview

This repository contains tools for benchmarking multi-agent systems on multiplication and other arithmetic tasks. The main components are:

- **multiagent_reasoner.py** - Multi-agent reasoning system with decomposition, voting, and composition
- **linear_multiagent_reasoner.py** - Linear variant of the multi-agent reasoner
- **agent_benchmark_runner.py** - Generic benchmark runner supporting parallel execution
- **make_synthetic_benchmarks.py** - Tool for generating synthetic benchmark datasets

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Single Problem

Test the multi-agent reasoner on a single multiplication problem:

```bash
LOG_LEVEL=INFO echo "What is 46048 × 42098?" | python multiagent_reasoner.py
```

### Benchmark Suite

Run the benchmark on a dataset:

```bash
python agent_benchmark_runner.py \
  --python-prog multiagent_reasoner.py \
  --task mul_5x5 \
  --local-jsonl data/bench_long_mul_5_5__200.jsonl \
  --answer-format number \
  --final-token '>>>>' \
  --num-workers 10 \
  --limit 10
```

### Configuration

The multi-agent reasoner supports several environment variables:

- `WINNING_VOTE_COUNT` - Number of votes required for consensus (default: 2)
- `MAX_DEPTH` - Maximum recursion depth for decomposition (default: 5)
- `LOG_LEVEL` - Logging level (default: INFO)
- `LOG_DIR` - Directory for detailed per-run logs (optional)
- `LOG_FAILURES_JSONL` - Path to log failure cases for analysis (optional)

Example with custom configuration:

```bash
export WINNING_VOTE_COUNT=8
export MAX_DEPTH=3
export LOG_FAILURES_JSONL="./failures.jsonl"
export LOG_DIR="./logs"

python agent_benchmark_runner.py \
  --python-prog multiagent_reasoner.py \
  --task mul_5x5 \
  --local-jsonl data/bench_long_mul_5_5__200.jsonl \
  --answer-format number \
  --final-token '>>>>' \
  --num-workers 50 \
  --limit 200 \
  --timeout-ms 180000
```

## Benchmark Runner Options

- `--python-prog` - Path to the Python program to benchmark
- `--task` - Task name (e.g., mul_5x5, gsm8k)
- `--local-jsonl` - Path to local JSONL benchmark file
- `--answer-format` - How to parse answers (number, list-json)
- `--final-token` - Token marking the final answer (default: ####)
- `--num-workers` - Number of parallel workers (default: 1)
- `--limit` - Limit number of problems to evaluate
- `--timeout-ms` - Per-item timeout in milliseconds (default: 120000)
- `--sample-retries` - Max retries per sample on failure (default: 2)

## Output

The benchmark runner produces:
- `results_{task}_{timestamp}.csv` - Detailed results for each problem
- `results_{task}_{timestamp}.jsonl` - Results in JSONL format
- `results_progress_{timestamp}.csv` - Real-time progress tracking

If failure logging is enabled:
- Failure logs in the specified JSONL file
- Per-run detailed logs in the LOG_DIR directory

## Instrumentation

See [INSTRUMENTATION_GUIDE.md](INSTRUMENTATION_GUIDE.md) for details on:
- Failure pattern analysis
- Trace data collection
- Debugging techniques

## Benchmark Data

The `data/` directory contains sample benchmark datasets:
- `bench_long_mul_5_5__200.jsonl` - 200 5×5 digit multiplication problems
- `bench_long_mul_10_10__200.jsonl` - 200 10×10 digit multiplication problems
- `bench_sort_len_500__50.jsonl` - 50 sorting problems

## License

See LICENSE.txt in the parent repository.

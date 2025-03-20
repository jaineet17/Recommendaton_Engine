# Load Testing the Recommendation Engine

This document provides instructions on how to use the load testing script `load_test_amazon_100k.py` to evaluate the performance and scalability of the recommendation engine.

## Prerequisites

Before running the load test, ensure you have:

1. The recommendation API server running (`python run_api.py`)
2. Required Python packages installed:
   - requests
   - numpy
   - pandas
   - matplotlib
   - scipy
   - kafka-python (optional)
   - prometheus-client (optional)
   - psutil
   - tqdm

## Basic Usage

To run a basic load test:

```bash
python load_test_amazon_100k.py --users 1000 --products 500 --interactions 10000 --concurrency 20 --duration 60
```

## Parameters

The load test script supports various parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--api-url` | Base URL for the API | http://localhost:5050/api |
| `--users` | Number of users to generate | 100,000 |
| `--products` | Number of products to generate | 50,000 |
| `--interactions` | Number of interactions to generate | 1,000,000 |
| `--concurrency` | Number of concurrent requests | 50 |
| `--duration` | Duration of load test in seconds | 120 |
| `--batch-size` | Batch size for processing | 1,000 |
| `--output-dir` | Output directory for test results | load_test_results |
| `--kafka-broker` | Kafka broker address (if applicable) | localhost:9092 |
| `--prometheus-url` | Prometheus server URL (if applicable) | http://localhost:9090 |
| `--prometheus-metrics-port` | Prometheus metrics port | 8001 |
| `--grafana-url` | Grafana URL (if applicable) | http://localhost:3000 |
| `--jenkins-url` | Jenkins URL (if applicable) | http://localhost:8080 |
| `--amazon-data` | Path to Amazon review dataset | (optional) |

## Examples

### Small Test (Development)

```bash
python load_test_amazon_100k.py --users 100 --products 50 --interactions 1000 --concurrency 10 --duration 30
```

### Medium Test (Testing)

```bash
python load_test_amazon_100k.py --users 5000 --products 1000 --interactions 50000 --concurrency 20 --duration 120
```

### Large Test (Production Validation)

```bash
python load_test_amazon_100k.py --users 100000 --products 10000 --interactions 1000000 --concurrency 50 --duration 300 --batch-size 5000
```

### Background Execution

For long-running tests, you can run the script in the background:

```bash
nohup python load_test_amazon_100k.py --users 100000 --products 10000 --interactions 1000000 --concurrency 50 --duration 300 --batch-size 5000 > load_test_100k.log 2>&1 &
```

## Test Results

The load test generates the following outputs in the specified output directory:

1. **Log file**: Contains detailed information about the test execution
2. **Performance charts**: Visual representation of response times, throughput, etc.
3. **Test summary CSV**: Tabular data with all metrics
4. **Response times CSV**: Detailed response time data

Key metrics reported include:

- Total requests
- Successful requests
- Success rate
- Throughput (requests/second)
- Average response time
- 95th percentile response time
- Component health status
- Recommendation quality metrics (diversity, coverage, personalization)
- System resource usage (CPU, memory)

## Testing the Feedback Loop

The script also tests the recommendation engine's feedback loop by:

1. Getting initial recommendations for a test user
2. Sending a purchase event for a specific product
3. Triggering the recommendation update
4. Getting updated recommendations
5. Checking if the recommendations have changed
6. Verifying that related products appear in the recommendations

This ensures that user interactions properly influence future recommendations.

## Monitoring Test Progress

You can monitor test progress in real time:

```bash
tail -f load_test_results/load_test.log
```

Or check test results after completion:

```bash
cat load_test_results/test_summary.csv
```

## Notes on Large-Scale Testing

When running large-scale tests (100k+ users):

1. Increase the batch size to efficiently process users (e.g., --batch-size 5000)
2. Allow sufficient duration for the test (e.g., --duration 300 or more)
3. Adjust concurrency based on your server capacity
4. Run in the background to avoid terminal interruptions
5. Monitor system resource usage during the test
6. Consider testing in a production-like environment 
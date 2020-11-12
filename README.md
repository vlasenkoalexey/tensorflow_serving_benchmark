# TensorFlow Serving benchmark
Tool for benchmarking TensorFlow Serving.

## Usage

```sh
# Install dependencies
pip3 install tensorflow
pip3 install tensorflow-serving-api
pip3 install pandas
pip3 install matplotlib

# Start model server in another terminal
docker run -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=...,target=/models/default \
  -e MODEL_NAME=default -t tensorflow/serving

# Run benchmark
python3 benchmark.py --port=8500 --mode='grpc' --tfrecord_dataset_path=...
```


## Flags
```
benchmark.py:
  --api_key: API Key for ESP service if authenticating external requests.
    (default: '')
  --batch_size: Per request batch size.
    (default: '8')
    (an integer)
  --csv_report_filename: Filename to generate report
    (default: '')
  --distribution: <uniform|poisson|pareto>: Distribution
    (default: 'uniform')
  --grpc_compression: <none|deflate|gzip>: gRPC compression
    (default: 'none')
  --host: Host name to connect to, localhost by default.
    (default: 'localhost')
  --input_name: The name of the model input tensor.
    (default: 'input')
  --mode: <grpc|sync_grpc|rest>: Benchmark mode: gRPC, synchronous gRPC, or REST
    (default: 'grpc')
  --model_name: Name of the model being served on the ModelServer
    (default: '')
  --num_requests: Total # of requests sent.
    (default: '20')
    (an integer)
  --num_warmup_requests: Number of warmup requests to send before benchmark.
    (default: '0')
    (an integer)
  --port: Port to connect to.
    (an integer)
  --qps_range: Desired client side request QPS inone of the following formats:
        - qps - benchmark at one QPS
        - start, stop - benchmark at QPS range [start, stop)
        - start, stop, step - benchmark at QPS range [start, stop) with step
        - [qps1, qps2] - benchmark at give QPS range values
    (default: '')
  --request_timeout: Timeout for inference request.
    (default: '300.0')
    (a number)
  --requests_file_path: The path to the requests file in json format.
    (default: '')
  --signature_name: Name of the model signature on the ModelServer
    (default: 'serving_default')
  --tfrecord_dataset_path: The path to data in tfrecord or tfrecord.gz format.
    (default: '')
  --workers: Number of workers.
    (default: '1')
    (an integer)
```

## Usage notes
- You can specify either `requests_file_path` or `tfrecord_dataset_path` argument to load data.
- If there are not enough records in input files, tool will loop over existing requests.
- Requests are sent asynchronously, if tool can't send requests at specified QPS rate, it will report average request miss rate. You can use more workers to workaround it and benchmark at higher QPS.
- If you specify `csv_report_filename` tool will also generate graph with latency distribution like this one:
![sample graph](./sample_report.csv.png)
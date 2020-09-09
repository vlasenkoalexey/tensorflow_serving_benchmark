# TensorFlow Serving benchmark
Tool for benchmarking TensorFlow Serving

## Usage

```sh
pip3 install tensorflow
pip3 install tensorflow-serving-api
python3 benchmark.py --tfrecord_dataset_path=...
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
  --requests_file_path: The path to requests file in json format.
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
You can specify either `requests_file_path` or `tfrecord_dataset_path` argument to load data.
`requests_file_path` p
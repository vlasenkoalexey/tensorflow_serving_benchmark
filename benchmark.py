"""Inference benchmark tool for TensorFlow Serving and Triton."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv

from requests.sessions import default_headers
import distribution
import functools

import io
import numbers
import os
import pandas as pd
import queue as Queue
import requests as r
import threading
import multiprocessing
import time
import numpy as np

# Disable GPU, so tensorflow initializes faster
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf

from google.protobuf.json_format import Parse as ProtoParseJson
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import gfile
from itertools import cycle, islice
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from clients import base_client

tf.app.flags.DEFINE_integer(
    "num_requests",
    0,
    "Total # of requests to send for each benchmark. "
    "Either num_requests or num_seconds has to be specified.")
tf.app.flags.DEFINE_float(
    "num_seconds",
    0,
    "Number of seconds to run each benchmark for. "
    "Either num_requests or num_seconds has to be specified.")
tf.app.flags.DEFINE_integer(
    "num_warmup_requests", 0,
    "Number requests to send before starting benchmark.")
tf.app.flags.DEFINE_integer(
    "num_warmup_delay_seconds", 2,
    "Number of seconds to wait after warmup requests are sent.")
tf.app.flags.DEFINE_string(
    "qps_range",
    "",
    "Desired client side request QPS in"
    "one of the following formats:"
    " - qps - benchmark at one QPS"
    " - start, stop - benchmark at QPS range [start, stop)"
    " - start, stop, step - benchmark at QPS range [start, stop) with step"
    " - [qps1, qps2] - benchmark at give QPS range values",
)
tf.app.flags.DEFINE_float("request_timeout", 300.0,
                          "Timeout for inference request.")
tf.app.flags.DEFINE_string("model_name", "",
                           "Name of the model being served on the ModelServer")
tf.app.flags.DEFINE_string("model_version", "",
                           "Version the model being served on the ModelServer")
tf.app.flags.DEFINE_string(
    "signature_name",
    "serving_default",
    "Name of the model signature on the ModelServer",
)
tf.app.flags.DEFINE_string(
    "host", "localhost",
    "Host name to connect to, localhost by default. Alternatively you can specify"
    "full URL to send requests to.")
tf.app.flags.DEFINE_integer("port", None, "Port to connect to.")
tf.app.flags.DEFINE_enum(
    "mode",
    "grpc",
    ["grpc", "sync_grpc", "rest", "triton_grpc", "triton_rest", "vertex_gapic", 
     "vertex_rest"],
    "Benchmark mode: gRPC, synchronous gRPC, or REST, or Triton format.",
)
tf.app.flags.DEFINE_enum("distribution", "uniform",
                         ["uniform", "poisson", "pareto"], "Distribution")
tf.app.flags.DEFINE_string(
    "tfrecord_dataset_path", "",
    "The path to data in tfrecord or tfrecord.gz format.")
tf.app.flags.DEFINE_string(
    "requests_file_path",
    "",
    "The path the predict_pb2.PredictRequest requests file serialized in json format.",
)
tf.app.flags.DEFINE_string("jsonl_file_path", "",
                           "The path the dataset file in jsonl format.")
tf.app.flags.DEFINE_string("input_name", "input",
                           "The name of the model input tensor.")
tf.app.flags.DEFINE_integer("batch_size", None, "Per request batch size.")
tf.app.flags.DEFINE_integer("workers", 1, "Number of workers.")
tf.app.flags.DEFINE_string(
    "api_key", "",
    "API Key for ESP service if authenticating external requests.")
tf.app.flags.DEFINE_string("csv_report_filename", "",
                           "Optional filename to generate report")
tf.app.flags.DEFINE_string("title", "", "Optional title for the report")
tf.app.flags.DEFINE_enum("grpc_compression", "none",
                         ["none", "deflate", "gzip"], "grpc compression")
tf.app.flags.DEFINE_string("authorization_header", "",
                           "Authorization header for REST requests.")
tf.app.flags.DEFINE_string("grpc_destination", "",
                           "gRPC destination metadata header.")
tf.app.flags.DEFINE_string("default_int_type", "",
                           "Default type to use for integer values.")
tf.app.flags.DEFINE_string("default_float_type", "",
                           "Default type to use for fractional values.")

FLAGS = tf.app.flags.FLAGS


def get_client_class():
  if FLAGS.mode == "grpc":
    from clients import tensorflow_serving_grpc
    return tensorflow_serving_grpc.TensorflowServingGrpc
  elif FLAGS.mode == "sync_grpc":
    from clients import tensorflow_serving_sync_grpc
    return tensorflow_serving_sync_grpc.TensorflowServingSyncGrpc
  elif FLAGS.mode == "rest":
    from clients import tensorflow_serving_rest
    return tensorflow_serving_rest.TensorflowServingRest
  elif FLAGS.mode == "triton_rest":
    from clients import triton_rest
    return triton_rest.TritonRest
  elif FLAGS.mode == "triton_grpc":
    from clients import triton_grpc
    return triton_grpc.TritonGrpc
  elif FLAGS.mode == "vertex_gapic":
    from clients import vertex_gapic
    return vertex_gapic.VertexGapic
  elif FLAGS.mode == "vertex_rest":
    from clients import vertex_rest
    return vertex_rest.VertexRest
  else:
    raise ValueError("Invalid mode")


def get_grpc_compression():
  if FLAGS.grpc_compression == "gzip":
    return grpc.Compression.Gzip
  elif FLAGS.grpc_compression == "deflate":
    return grpc.Compression.Deflate
  else:
    return None


def get_qps_range(qps_range_string):
  qps_range_string = qps_range_string.strip()
  if qps_range_string.startswith("[") and qps_range_string.endswith("]"):
    qps_range_string = qps_range_string.lstrip("[").rstrip("]")
    qps_range_list = list(map(lambda v: float(v), qps_range_string.split(",")))
    return qps_range_list

  qps_range_list = list(map(lambda v: float(v), qps_range_string.split(",")))
  qps_range_start = 0
  qps_range_step = 1
  if len(qps_range_list) == 1:
    return [qps_range_list[0]]
  elif len(qps_range_list) == 2:
    return list(np.arange(qps_range_list[0], qps_range_list[1]))
  elif len(qps_range_list) == 3:
    return list(np.arange(qps_range_list[0], qps_range_list[1], qps_range_list[2]))
  else:
    raise ValueError("Invalid argument qps_range:" + qps_range_start)


def merge_results(results, result):
  for key, value in result.items():
    if not key.startswith("_"):
      if key not in results:
        results[key] = [value]
      else:
        results[key].append(value)


def merge_worker_results(worker_results):
  success = 0
  error = 0
  reqested_qps = 0
  start_time = []
  end_time = []
  latency = []
  avg_miss_rate_percent = []
  for worker_result in worker_results:
    success += worker_result["success"]
    error += worker_result["error"]
    reqested_qps += worker_result["reqested_qps"]
    avg_miss_rate_percent.append(worker_result["avg_miss_rate_percent"])
    latency.extend(worker_result["_latency"])
    start_time.append(worker_result["_start_time"])
    end_time.append(worker_result["_end_time"])

  time = np.max(end_time) - np.min(start_time)
  return {
      "reqested_qps": reqested_qps,
      "actual_qps": (success + error) / time,
      "success": success,
      "error": error,
      "time": time,
      "avg_latency": np.average(latency) * 1000,
      "p50": np.percentile(latency, 50) * 1000,
      "p90": np.percentile(latency, 90) * 1000,
      "p99": np.percentile(latency, 99) * 1000,
      "avg_miss_rate_percent": np.average(avg_miss_rate_percent),
  }


def print_result(result):
  v = []
  for key, value in result.items():
    if not key.startswith("_"):
      if "float" in str(type(value)):
        v.append("{}: {:.2f}".format(key, value))
      else:
        v.append("{}: {}".format(key, value))
  tf.logging.info("\t".join(v))


def get_requests_for_qps(requests_list, qps, num_requests=None, num_seconds=None, worker_index=0):
  if num_requests:
    pass
  elif num_seconds:
    num_requests = int(max(qps * num_seconds, 10))
  else:
    raise ValueError("Neither num_requests, nor num_seconds are specified")
  print('num_requests', num_requests)
  return islice(cycle(requests_list),
                worker_index * num_requests,
                (worker_index + 1) * num_requests), num_requests


def main(argv):
  del argv
  tf.disable_v2_behavior()

  if FLAGS.qps_range is None or FLAGS.qps_range == "":
    tf.logging.error("Please specify qps_range")
    exit(1)

  if not FLAGS.num_requests and not FLAGS.num_seconds:
    tf.logging.error("Either num_requests, or num_seconds should be specified")

  request_path = None
  request_format = None
  if FLAGS.tfrecord_dataset_path != "":
    request_format = base_client.RequestFormat.TFRECORD
    request_path = FLAGS.tfrecord_dataset_path
  elif FLAGS.requests_file_path != "":
    request_format = base_client.RequestFormat.FILE
    request_path = FLAGS.requests_file_path
  elif FLAGS.jsonl_file_path != "":
    request_format = base_client.RequestFormat.DICTIONARY
    request_path = FLAGS.jsonl_file_path
  else:
    raise ValueError(
        "Either tfrecord_dataset_path or requests_file_path flag has to be specified"
    )

  http_headers = {}
  if FLAGS.authorization_header:
    http_headers["authorization"] = FLAGS.authorization_header

  grpc_metadata = []
  if FLAGS.api_key:
    grpc_metadata.append(("x-api-key", FLAGS.api_key))
  if FLAGS.grpc_destination:
    grpc_metadata.append(("grpc-destination", FLAGS.grpc_destination))

  client_class = get_client_class()
  client = client_class(FLAGS.host, FLAGS.port, FLAGS.model_name,
                        FLAGS.model_version, FLAGS.signature_name,
                        FLAGS.distribution, FLAGS.input_name,
                        FLAGS.default_int_type,
                        FLAGS.default_float_type, http_headers, grpc_metadata,
                        get_grpc_compression(), FLAGS.request_timeout)

  tf.logging.info("Loading data")
  requests_list = client.get_requests(request_format, request_path,
                                      FLAGS.num_warmup_requests,
                                      FLAGS.batch_size)

  results = {}
  if FLAGS.num_warmup_requests > 0:
    tf.logging.info("Sending {} warmup requests".format(
        FLAGS.num_warmup_requests))
    warmup_qps = get_qps_range(FLAGS.qps_range)[0]
    warmup_requests, num_requests = get_requests_for_qps(
      requests_list, warmup_qps, num_requests=FLAGS.num_warmup_requests)
    _ = client.run(warmup_requests, num_requests, warmup_qps)
    if FLAGS.num_warmup_delay_seconds:
      tf.logging.info("Waiting for %d seconds after warmup", FLAGS.num_warmup_delay_seconds)
      time.sleep(FLAGS.num_warmup_delay_seconds)
    tf.logging.info("Warmup complete")

  if FLAGS.workers == 1:
    for qps in get_qps_range(FLAGS.qps_range):
      worker_requests, num_requests = get_requests_for_qps(
        requests_list,
        qps,
        FLAGS.num_requests,
        FLAGS.num_seconds)
      result = client.run(worker_requests, num_requests, qps)
      print_result(result)
      merge_results(results, result)
  else:

    def _worker_load_test_func(qps, worker_results, worker_index):
      worker_requests, num_requests = get_requests_for_qps(
        requests_list,
        qps,
        FLAGS.num_requests,
        FLAGS.num_seconds,
        worker_index = worker_index)
      worker_results[worker_index] = client.run(worker_requests,
                                                num_requests, qps)

    for qps in get_qps_range(FLAGS.qps_range):
      worker_processes = []
      with multiprocessing.Manager() as manager:
        worker_results = manager.list()
        for worker_index in range(FLAGS.workers):
          worker_process = multiprocessing.Process(
              target=_worker_load_test_func,
              args=(qps, worker_results, worker_index))
          worker_processes.append(worker_process)
          worker_results.append({})

        for worker_process in worker_processes:
          # Offset each worker's startup time.
          # Otherwise, the uniform distribution would become a uniform-burst.
          time.sleep(1 / qps / len(worker_processes))
          worker_process.start()

        for worker_process in worker_processes:
          worker_process.join()

        result = merge_worker_results(worker_results)
        print_result(result)
        merge_results(results, result)

  if FLAGS.title and "reqested_qps" in results and len(
      results["reqested_qps"]) > 0:
    results["title"] = [""] * len(results["reqested_qps"])
    results["title"][0] = FLAGS.title

  df = pd.DataFrame.from_dict(results)
  tf.logging.info("\n" + df.to_string(
      columns=[
          "reqested_qps", "actual_qps", "success", "error", "avg_latency",
          "p99", "avg_miss_rate_percent"
      ],
      index=False))

  if FLAGS.csv_report_filename is not None and FLAGS.csv_report_filename != "":
    df.to_csv(FLAGS.csv_report_filename)

    import matplotlib.pyplot as plt

    base_image_file_name = FLAGS.csv_report_filename.replace(".csv",
                                                             "").replace(
                                                                 ".", "_")
    fig, (p1, p2) = plt.subplots(2, 1, figsize=(10, 12), facecolor=(1, 1, 1))
    p1.set_title(FLAGS.title + " Requested QPS")
    p1.plot("reqested_qps", "p50", data=results, label="p50")
    p1.plot("reqested_qps", "p90", data=results, label="p90")
    p1.plot("reqested_qps", "p99", data=results, label="p99")
    p1.plot("reqested_qps", "avg_latency", data=results, label="avg_latency")
    p1.legend()

    plt.title(FLAGS.title + " Actual QPS")
    p2.plot("actual_qps", "p50", data=results, label="p50")
    p2.plot("actual_qps", "p90", data=results, label="p90")
    p2.plot("actual_qps", "p99", data=results, label="p99")
    p2.plot("actual_qps", "avg_latency", data=results, label="avg_latency")
    p2.legend()

    plt.savefig(base_image_file_name + ".png")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  logger = tf.get_logger()
  logger.propagate = False
  tf.app.run(main)

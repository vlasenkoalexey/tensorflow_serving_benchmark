"""A loadtest script which sends request via GRPC to TF inference server.
Adapted from:
https://github.com/tensorflow/tpu/blob/master/models/experimental/inference/load_test_client.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import distribution
import functools
import grpc
import io
import json
import numpy as np
import os
import pandas as pd
import queue as Queue
import requests as r
import threading
import multiprocessing
import time

# Disable GPU, so tensorflow initializes faster
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf

from google.protobuf.json_format import Parse as ProtoParseJson
from tensorflow.python.platform import gfile
from itertools import cycle, islice
from protos.tensorflow.core.framework import types_pb2
from protos.tensorflow_serving.apis import predict_pb2
from protos.tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer("num_requests", 20, "Total # of requests sent.")
tf.app.flags.DEFINE_integer(
    "num_warmup_requests", 0,
    "Number requests to send before starting benchmark.")
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
tf.app.flags.DEFINE_string(
    "model_name", "", "Name of the model being served on the ModelServer")
tf.app.flags.DEFINE_string(
    "signature_name",
    "serving_default",
    "Name of the model signature on the ModelServer",
)
tf.app.flags.DEFINE_string("host", "localhost",
                           "Host name to connect to, localhost by default.")
tf.app.flags.DEFINE_integer("port", None, "Port to connect to.")
tf.app.flags.DEFINE_enum(
    "mode",
    "grpc",
    ["grpc", "sync_grpc", "rest"],
    "Benchmark mode: gRPC, synchronous gRPC, or REST",
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
                           "Filename to generate report")
tf.app.flags.DEFINE_enum("grpc_compression", "none",
                         ["none", "deflate", "gzip"], "grpc compression")

FLAGS = tf.app.flags.FLAGS


class Worker(object):
  """A loadtest worker which sends RPC request."""

  __slot__ = (
      "_id",
      "_request",
      "_stub",
      "_queue",
      "_success",
      "_start_time",
      "_end_time",
      "_qps",
      "_num_requests",
      "_metadata",
  )

  def __init__(self, index, request, stub, queue, qps, num_requests,
               error_details, metadata):
    self._id = index
    self._request = request
    self._stub = stub
    self._queue = queue
    self._qps = qps
    self._num_requests = num_requests
    self._success = None
    self._start_time = None
    self._end_time = None
    self._error_details = error_details
    self._metadata = metadata

  def start(self):
    """Start to send request."""
    def _callback(resp_future):
      """Callback for aynchronous inference request sent."""
      exception = resp_future.exception()
      if exception:
        self._success = False
        if hasattr(exception, 'details'):
          if not exception.details() in self._error_details:
            self._error_details.add(exception.details())
            tf.logging.error(exception)
        else:
          tf.logging.error(exception)
      else:
        self._success = True
      self._end_time = time.time()
      self._queue.get()
      self._queue.task_done()
      processed_count = self._num_requests - self._queue.qsize()
      if processed_count % (10 * self._qps) == 0:
        tf.logging.debug("received {} responses".format(processed_count))

    def _send_rpc():
      self._start_time = time.time()
      resp_future = self._stub.Predict.future(self._request,
                                              FLAGS.request_timeout,
                                              metadata=self._metadata)
      resp_future.add_done_callback(_callback)

    _send_rpc()

  def cancel(self):
    self._rpc.StartCancel()

  @property
  def success_count(self):
    return int(self._success)

  @property
  def error_count(self):
    return int(not self._success)

  @property
  def latency(self):
    if not (self._start_time and self._end_time):
      raise Exception("Request is not complete yet.")
    return self._end_time - self._start_time


def get_grpc_compression():
  if FLAGS.grpc_compression == 'gzip':
    return grpc.Compression.Gzip
  elif FLAGS.grpc_compression == 'deflate':
    return grpc.Compression.Deflate
  else:
    return None


def run_grpc_load_test(address, requests, num_requests, qps):
  """Loadtest the server gRPC endpoint with constant QPS.
    Args:
      address: The model server address to which send inference requests.
      requests: Iterable of PredictRequest proto.
      num_requests: Number of requests.
      qps: The number of requests being sent per second.
    """

  tf.logging.info("Running gRPC benchmark at {} qps".format(qps))

  grpc_channel = grpc.insecure_channel(address,
                                       compression=get_grpc_compression())
  stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)

  dist = distribution.Distribution.factory(FLAGS.distribution, qps)
  metadata = []
  if FLAGS.api_key:
    metadata.append(("x-api-key", FLAGS.api_key))

  q = Queue.Queue()
  intervals = []

  for i in range(num_requests):
    q.put(i)
    intervals.append(dist.next())
    i = i + 1

  i = 0
  workers = []
  miss_rate_percent = []
  start_time = time.time()
  previous_worker_start = start_time
  error_details = set()
  for request in requests:
    interval = intervals[i]
    worker = Worker(i, request, stub, q, qps, num_requests, error_details,
                    metadata)
    workers.append(worker)
    worker.start()

    if i % (qps * 10) == 0:
      tf.logging.debug("sent {} requests.".format(i))
    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = (100 *
                      ((time.time() - previous_worker_start) - interval) /
                      (interval))
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()
    i = i + 1

  # block until all workers are done
  q.join()
  acc_time = time.time() - start_time
  success_count = 0
  error_count = 0
  latency = []
  worker_end_time = start_time
  for w in workers:
    success_count += w.success_count
    error_count += w.error_count
    latency.append(w.latency)
    worker_end_time = (w._end_time
                       if w._end_time > worker_end_time else worker_end_time)

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn(
        "couldn't keep up at current QPS rate, average miss rate:{:.2f}%".
        format(avg_miss_rate_percent))

  return {
      "reqested_qps": qps,
      "actual_qps": num_requests / acc_time,
      "success": success_count,
      "error": error_count,
      "time": acc_time,
      "avg_latency": np.average(latency) * 1000,
      "p50": np.percentile(latency, 50) * 1000,
      "p90": np.percentile(latency, 90) * 1000,
      "p99": np.percentile(latency, 99) * 1000,
      "avg_miss_rate_percent": avg_miss_rate_percent,
      "_latency": latency,
      "_start_time": start_time,
      "_end_time": time.time(),
  }


def run_synchronous_grpc_load_test(address, requests, num_requests, qps):
  """Loadtest the server gRPC endpoint with constant QPS.

    This API is sending gRPC requests in synchronous mode,
    one request per Thread.
    Args:
      address: The model server address to which send inference requests.
      requests: Iterable of PredictRequest proto.
      num_requests: Number of requests.
      qps: The number of requests being sent per second.
    """

  tf.logging.info("Running synchronous gRPC benchmark at {} qps".format(qps))

  grpc_channel = grpc.insecure_channel(address,
                                       compression=get_grpc_compression())
  stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)

  # List appends are thread safe
  success = []
  error = []
  latency = []
  metadata = []
  error_details = set()
  if FLAGS.api_key:
    metadata.append(("x-api-key", FLAGS.api_key))
  dist = distribution.Distribution.factory(FLAGS.distribution, qps)

  def _make_grpc_call(i, request):
    """Send GRPC request to Tensorflow Serving endpoint."""
    start_time = time.time()
    try:
      resp = stub.Predict(request, FLAGS.request_timeout, metadata=metadata)
      success.append(1)
    except Exception as e:
      error.append(1)
      if hasattr(e, 'details'):
        if not e.details() in error_details:
          error_details.add(e.details())
          tf.logging.error(e)
      else:
        tf.logging.error(e)

    latency.append(time.time() - start_time)
    if len(latency) % (qps * 10) == 0:
      tf.logging.debug("received {} responses.".format(len(latency)))

  intervals = []
  for i in range(num_requests):
    intervals.append(dist.next())

  thread_lst = []
  miss_rate_percent = []
  start_time = time.time()
  previous_worker_start = start_time
  i = 0
  for request in requests:
    interval = intervals[i]
    thread = threading.Thread(target=_make_grpc_call, args=(i, request))
    thread_lst.append(thread)
    thread.start()
    if i % (qps * 10) == 0:
      tf.logging.debug("sent {} requests.".format(i))

    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = (100 *
                      ((time.time() - previous_worker_start) - interval) /
                      (interval))
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()
    i = i + 1

  for thread in thread_lst:
    thread.join()

  acc_time = time.time() - start_time

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn(
        "couldn't keep up at current QPS rate, average miss rate:{:.2f}%".
        format(avg_miss_rate_percent))

  return {
      "reqested_qps": qps,
      "actual_qps": num_requests / acc_time,
      "success": sum(success),
      "error": sum(error),
      "time": acc_time,
      "avg_latency": np.average(latency) * 1000,
      "p50": np.percentile(latency, 50) * 1000,
      "p90": np.percentile(latency, 90) * 1000,
      "p99": np.percentile(latency, 99) * 1000,
      "avg_miss_rate_percent": avg_miss_rate_percent,
      "_latency": latency,
      "_start_time": start_time,
      "_end_time": time.time(),
  }


def run_rest_load_test(address, requests, num_requests, qps):
  """Loadtest the server REST endpoint with constant QPS.

    Args:
      address: The model server address to which send inference requests.
      requests: Iterable of POST request bodies.
      num_requests: Number of requests.
      qps: The number of requests being sent per second.
    """

  tf.logging.info("Running REST benchmark at {} qps".format(qps))

  address = "http://{}/v1/models/{}:predict".format(address, FLAGS.model_name)
  dist = distribution.Distribution.factory(FLAGS.distribution, qps)

  # List appends are thread safe
  success = []
  error = []
  latency = []

  def _make_rest_call(i, request):
    """Send REST POST request to Tensorflow Serving endpoint."""
    start_time = time.time()
    resp = r.post(address, data=request)
    latency.append(time.time() - start_time)
    if len(latency) % (10 * qps) == 0:
      tf.logging.debug("received {} responses.".format(len(latency)))
    if resp.status_code == 200:
      success.append(1)
    else:
      tf.logging.error(resp.json())
      error.append(1)
    resp.close()

  intervals = []
  for i in range(num_requests):
    intervals.append(dist.next())

  thread_lst = []
  miss_rate_percent = []
  start_time = time.time()
  previous_worker_start = start_time
  i = 0
  for request in requests:
    interval = intervals[i]
    thread = threading.Thread(target=_make_rest_call, args=(i, request))
    thread_lst.append(thread)
    thread.start()
    if i % (10 * qps) == 0:
      tf.logging.debug("sent {} requests.".format(i))

    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = (100 *
                      ((time.time() - previous_worker_start) - interval) /
                      (interval))
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()
    i = i + 1

  for thread in thread_lst:
    thread.join()

  acc_time = time.time() - start_time

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn(
        "couldn't keep up at current QPS rate, average miss rate:{:.2f}%".
        format(avg_miss_rate_percent))

  return {
      "reqested_qps": qps,
      "actual_qps": num_requests / acc_time,
      "success": sum(success),
      "error": sum(error),
      "time": acc_time,
      "avg_latency": np.average(latency) * 1000,
      "p50": np.percentile(latency, 50) * 1000,
      "p90": np.percentile(latency, 90) * 1000,
      "p99": np.percentile(latency, 99) * 1000,
      "avg_miss_rate_percent": avg_miss_rate_percent,
      "_latency": latency,
      "_start_time": start_time,
      "_end_time": time.time(),
  }


def get_rows(path, count):
  tf.logging.info("loading data for prediction")
  compression_type = "GZIP" if path.endswith(".gz") else None
  dataset = tf.data.TFRecordDataset(gfile.Glob(path),
                                    compression_type=compression_type)
  if FLAGS.batch_size is not None:
    dataset = dataset.batch(FLAGS.batch_size)
  rows = []
  itr = tf.data.make_initializable_iterator(dataset.repeat())
  next = itr.get_next()
  with tf.Session() as sess:
    sess.run(itr.initializer)
    for _ in range(count):
      inputs = sess.run(next)
      rows.append(inputs)
  return rows


def generate_rest_request(tfrecord_row):
  """Generate REST inference request's payload."""

  examples = [{"b64": base64.b64encode(row).decode()} for row in tfrecord_row]

  payload = json.dumps({
      "signature_name": FLAGS.signature_name,
      "inputs": {
          FLAGS.input_name: examples
      },
  })
  return payload


def generate_grpc_request(tfrecord_row):
  """Generate gRPC inference request with payload."""

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name
  request.inputs[FLAGS.input_name].CopyFrom(
      tf.make_tensor_proto(tfrecord_row, dtype=tf.string))
  return request

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2


def get_dimensions(list_of_lists):
    """Returns the inferred dense dimensions of a list of lists."""
    if not isinstance(list_of_lists, (list, tuple)):
        return []
    elif not list_of_lists:
        return [0]
    else:
        return [len(list_of_lists)] + get_dimensions(list_of_lists[0])

def make_tensor_proto(values, dtype=None):
    nparray = np.array(values)
    dims = get_dimensions(values)
    tensor_proto = tensor_pb2.TensorProto(
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
                dim=[
                    tensor_shape_pb2.TensorShapeProto.Dim(
                        size=d
                    )
                    for d in dims
                ],
        )
    )
    if not dtype:
        if len(dims) > 1:
            numpy_dtype = np.concatenate(nparray).ravel().dtype
        else:
            numpy_dtype = nparray.dtype
        if numpy_dtype == np.dtype(np.int64) or numpy_dtype == np.dtype(np.int32):
            dtype = types_pb2.DT_INT64
        elif numpy_dtype == np.dtype(np.float64) or numpy_dtype == np.dtype(np.float32):
            dtype = types_pb2.DT_FLOAT
        elif numpy_dtype == 'S':
            dtype = types_pb2.DT_STRING
        else:
            raise "Don't know how to convert:" + str(nparray.dtype)

    tensor_proto.dtype = dtype

    if dtype == types_pb2.DT_INT64:
        if len(dims) > 1:
            tensor_proto.int64_val.extend([item for sublist in values for item in sublist])
        else:
            tensor_proto.int64_val.extend(values)
    elif dtype == types_pb2.DT_FLOAT:
        if len(dims) > 1:
            tensor_proto.float_val.extend([item for sublist in values for item in sublist])
        else:
            tensor_proto.float_val.extend(values)
    else:
        raise "Don't know how to convert:" + str(nparray.dtype)
    return tensor_proto

# def generate_grpc_request_from_dictionary(row_dict):
#   """Generate gRPC inference request with payload."""

#   request = predict_pb2.PredictRequest()
#   request.model_spec.name = FLAGS.model_name
#   request.model_spec.signature_name = FLAGS.signature_name
#   for key, value in row_dict.items():
#     proto = make_tensor_proto(value)
#     if proto.dtype == types_pb2.DT_FLOAT:
#       proto.tensor_content = b""
#       if len(proto.tensor_shape.dim) == 1:
#         proto.float_val.extend(value)
#       elif len(proto.tensor_shape.dim) > 1:
#         proto.float_val.extend([item for sublist in value for item in sublist])

#     if proto.dtype == types_pb2.DT_INT32 or "values" in key or proto.dtype == types_pb2.DT_INT64:
#       proto = make_tensor_proto(value, types_pb2.DT_INT64)
#       proto.tensor_content = b""
#       if len(proto.tensor_shape.dim) == 1:
#         proto.int64_val.extend(value)
#       elif len(proto.tensor_shape.dim) > 1:
#         proto.int64_val.extend([item for sublist in value for item in sublist])
#     #proto.tensor_content = b""
#     request.inputs[key].CopyFrom(proto)
#   return request

def generate_grpc_request_from_dictionary(row_dict):
  """Generate gRPC inference request with payload."""

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name
  for key, value in row_dict.items():
    proto = tf.make_tensor_proto(value)
    if proto.dtype == types_pb2.DT_INT32 or "values" in key:
      proto = tf.make_tensor_proto(value, types_pb2.DT_INT64)
    request.inputs[key].CopyFrom(proto)
  return request

def generate_rest_request_from_dictionary(row_dict):
  payload = json.dumps({
      "signature_name": FLAGS.signature_name,
      "inputs": row_dict
  })
  return payload

def get_requests():
  if FLAGS.tfrecord_dataset_path != "":
    rows = get_rows(FLAGS.tfrecord_dataset_path,
                    FLAGS.num_requests * FLAGS.workers)
    if FLAGS.mode == "grpc" or FLAGS.mode == "sync_grpc":
      return [generate_grpc_request(row) for row in rows]
    elif FLAGS.mode == "rest":
      return [generate_rest_request(row) for row in rows]
    else:
      raise ValueError("Invalid --mode:" + FLAGS.mode)
  elif FLAGS.requests_file_path != "":
    if FLAGS.mode == "grpc" or FLAGS.mode == "sync_grpc":
      with open(FLAGS.requests_file_path, "r") as f:
        j = json.load(f)
        if j is not list:
          j = [j]
        rows = [
            ProtoParseJson(json.dumps(row), predict_pb2.PredictRequest())
            for row in j
        ]
        return rows
    elif FLAGS.mode == "rest":
      with open(FLAGS.requests_file_path, "r") as f:
        j = json.load(f)
        if j is not list:
          j = [j]
        rows = [row for row in j]
        return rows
    else:
      raise ValueError("Invalid --mode:" + FLAGS.mode)
  elif FLAGS.jsonl_file_path != "":
    rows = []
    _generate_request_from_dictionary = None
    if FLAGS.mode == "grpc" or FLAGS.mode == "sync_grpc":
      _generate_request_from_dictionary = generate_grpc_request_from_dictionary
    elif FLAGS.mode == "rest":
      _generate_request_from_dictionary = generate_rest_request_from_dictionary
    else:
      raise ValueError("Invalid --mode:" + FLAGS.mode)

    with open(FLAGS.jsonl_file_path, "r") as f:
      for line in f:
        row_dict = eval(line)
        rows.append(_generate_request_from_dictionary(row_dict))
    return rows

  else:
    raise ValueError(
        "Either tfrecord_dataset_path or requests_file_path flag has to be specified"
    )


def get_qps_range(qps_range_string):
  qps_range_string = qps_range_string.strip()
  if qps_range_string.startswith("[") and qps_range_string.endswith("]"):
    qps_range_string = qps_range_string.lstrip("[").rstrip("]")
    qps_range_list = list(map(lambda v: int(v), qps_range_string.split(",")))
    return qps_range_list

  qps_range_list = list(map(lambda v: int(v), qps_range_string.split(",")))
  qps_range_start = 0
  qps_range_step = 1
  if len(qps_range_list) == 1:
    return [qps_range_list[0]]
  elif len(qps_range_list) == 2:
    return range(qps_range_list[0], qps_range_list[1])
  elif len(qps_range_list) == 3:
    return range(qps_range_list[0], qps_range_list[1], qps_range_list[2])
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


def main(argv):
  del argv
  tf.disable_v2_behavior()

  if FLAGS.qps_range is None or FLAGS.qps_range == "":
    tf.logging.error("Please specify qps_range")
    exit(1)

  address = "{}:{}".format(FLAGS.host, FLAGS.port)
  tf.logging.info("ModelServer at: {}".format(address))

  tf.logging.info("Loading data")
  requests_list = get_requests()

  if len(requests_list) < FLAGS.workers * FLAGS.num_requests:
    tf.logging.warn("Dataset you specified contains data for {} requests, "
                    "while you need {} requests for each of {} workers. "
                    "Some requests are going to be reused.".format(
                        len(requests_list), FLAGS.num_requests, FLAGS.workers))

  results = {}
  load_test_func = None

  if FLAGS.mode == "grpc" or FLAGS.mode == "sync_grpc":
    if FLAGS.mode == "grpc":
      load_test_func = functools.partial(run_grpc_load_test, address)
    else:
      load_test_func = functools.partial(run_synchronous_grpc_load_test,
                                         address)
  elif FLAGS.mode == "rest":
    load_test_func = functools.partial(run_rest_load_test, address)
  else:
    raise ValueError("Invalid --mode:" + FLAGS.mode)

  if FLAGS.num_warmup_requests > 0:
    tf.logging.info("Sending {} warmup requests".format(
        FLAGS.num_warmup_requests))
    warmup_requests = islice(cycle(requests_list), FLAGS.num_warmup_requests)
    _ = load_test_func(warmup_requests, FLAGS.num_warmup_requests, get_qps_range(FLAGS.qps_range)[0])
    tf.logging.info("Warmup complete")

  if FLAGS.workers == 1:
    for qps in get_qps_range(FLAGS.qps_range):
      worker_requests = islice(cycle(requests_list), FLAGS.num_requests)
      result = load_test_func(worker_requests, FLAGS.num_requests, qps)
      print_result(result)
      merge_results(results, result)
  else:

    def _worker_load_test_func(qps, worker_results, worker_index):
      worker_requests = islice(
          cycle(requests_list),
          worker_index * FLAGS.num_requests,
          (worker_index + 1) * FLAGS.num_requests,
      )
      worker_results[worker_index] = load_test_func(worker_requests, FLAGS.num_requests, qps)

    for qps in get_qps_range(FLAGS.qps_range):
      worker_processes = []
      with multiprocessing.Manager() as manager:
        worker_results = manager.list()
        for worker_index in range(FLAGS.workers):
          worker_process = multiprocessing.Process(target=_worker_load_test_func,
                                                   args=(qps, worker_results,
                                                         worker_index))
          worker_processes.append(worker_process)
          worker_results.append({})
          worker_process.start()

        for worker_process in worker_processes:
          worker_process.join()

        result = merge_worker_results(worker_results)
        print_result(result)
        merge_results(results, result)

  df = pd.DataFrame.from_dict(results)
  tf.logging.info("\n" + df.to_string(columns=['reqested_qps', 'actual_qps', 'success', 'error', 'avg_latency', 'p99', 'avg_miss_rate_percent'], index=False))

  if FLAGS.csv_report_filename is not None and FLAGS.csv_report_filename != "":
    df.to_csv(FLAGS.csv_report_filename)

    import matplotlib.pyplot as plt

    base_image_file_name = FLAGS.csv_report_filename.replace(".csv", "").replace(".", "_")

    plt.figure(figsize=(12, 8))
    plt.title("Requested QPS")
    plt.plot("reqested_qps", "p50", data=results, label="p50")
    plt.plot("reqested_qps", "p90", data=results, label="p90")
    plt.plot("reqested_qps", "p99", data=results, label="p99")
    plt.plot("reqested_qps", "avg_latency", data=results, label="avg_latency")
    plt.legend()
    plt.savefig(base_image_file_name + "_reqested_qps.png")

    plt.figure(figsize=(12, 8))
    plt.title("Actual QPS")
    plt.plot("actual_qps", "p50", data=results, label="p50")
    plt.plot("actual_qps", "p90", data=results, label="p90")
    plt.plot("actual_qps", "p99", data=results, label="p99")
    plt.plot("actual_qps", "avg_latency", data=results, label="avg_latency")
    plt.legend()
    plt.savefig(base_image_file_name + "_actual_qps.png")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)

"""Client for Triton Inference Server using gRPC API.

References:
- https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#grpc
- https://github.com/triton-inference-server/client/tree/master/src/python/examples
- https://github.com/triton-inference-server/client/blob/master/src/python/library/tritonclient/grpc/__init__.py
"""

import time
import clients.base_client
import distribution
import tensorflow.compat.v1 as tf
import threading
import queue as Queue
import grpc
import json
import numpy as np
from google.protobuf.json_format import Parse as ProtoParseJson
from tensorflow.python.platform import gfile
from tensorflow.core.framework import types_pb2
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tensorflow.python.framework import dtypes

import tritonclient.grpc as triton_grpcclient
import tritonclient.utils as triton_utils


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
               error_details, metadata, grpc_compression):
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
    self._grpc_compression = grpc_compression

  def start(self):
    """Start to send request."""

    def _callback(resp_future):
      """Callback for aynchronous inference request sent."""
      exception = resp_future.exception()
      if exception:
        self._success = False
        if hasattr(exception, "details"):
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
      resp_future = self._stub.ModelInfer.future(
          self._request, 300.0, metadata=self._metadata)
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


class TritonGrpc(clients.base_client.BaseClient):

  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()

  def generate_grpc_request_from_dictionary(self, row_dict):
    def isSubType(value, t):
      while isinstance(value, list):
        if len(value) > 0:
          value = value[0]
        else:
          return False
      return isinstance(value, t)

    def getType(value):
      if isSubType(value, float):
        return dtypes.as_dtype(self._default_float_type).as_numpy_dtype if self._default_float_type else numpy.float32
      elif isSubType(value, int):
        return dtypes.as_dtype(self._default_int_type).as_numpy_dtype if self._default_int_type else numpy.int32
      elif isSubType(value, str):
        return np.object_
      elif isSubType(value, bool):
        return np.bool
      else:
        raise ValueError("Can't detect type for:" + str(value))
        
    def map_multi_dimensional_list(l, transform):
        if type(l) == list and len(l) > 0:
            if type(l[0]) != list:
                return [transform(v) for v in l]
            else:
                return [map_multi_dimensional_list(v, transform) for v in l]
        else:
            return []

    triton_request_inputs = []
    for key, value in row_dict.items():
      t = getType(value)
      if t == np.object_:
        value = map_multi_dimensional_list(value, lambda s: s.encode("utf-8"))
      numpy_value = np.array(value, dtype=t)
      triton_request_input = triton_grpcclient.InferInput(
        key, 
        list(numpy_value.shape), 
        triton_utils.np_to_triton_dtype(t))
      triton_request_input.set_data_from_numpy(numpy_value)
      triton_request_inputs.append(triton_request_input)
    # https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/grpc/__init__.py#L64
    return triton_grpcclient._get_inference_request(
      model_name=self._model_name,
      inputs=triton_request_inputs,
      model_version=self._model_version,
      request_id="", 
      outputs=None,
      sequence_id=0,
      sequence_start=0,
      sequence_end=0,
      priority=0, 
      timeout=None)
  
  def get_requests_from_dictionary(self, path):
    rows = []
    with tf.gfile.GFile(path, "r") as f:
      for line in f:
        row_dict = eval(line)  # should it be json.loads??
        rows.append(self.generate_grpc_request_from_dictionary(row_dict))
    return rows

  def get_requests_from_file(self, path):
    with tf.gfile.GFile(path, "r") as f:
      j = json.load(f)
      if j is not list:
        j = [j]
      rows = [
          ProtoParseJson(json.dumps(row), service_pb2.ModelInferRequest())
          for row in j
      ]
      return rows

  def run(self, requests, num_requests, qps):
    """Runs the benchmark given address, requests and QPS.

      Args:
        requests: Iterable of POST request bodies.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """
    tf.logging.info("Running gRPC benchmark at {} qps".format(qps))

    address = f"{self._host}:{self._port}"

    grpc_channel = grpc.insecure_channel(
        address, compression=self._grpc_compression)
    stub = service_pb2_grpc.GRPCInferenceServiceStub(grpc_channel)

    dist = distribution.Distribution.factory(self._distribution, qps)

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
                      self._grpc_metadata, self._grpc_compression)
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
      worker_end_time = (
          w._end_time if w._end_time > worker_end_time else worker_end_time)

    avg_miss_rate_percent = 0
    if len(miss_rate_percent) > 0:
      avg_miss_rate_percent = np.average(miss_rate_percent)
      tf.logging.warn(
          "couldn't keep up at current QPS rate, average miss rate:{:.2f}%"
          .format(avg_miss_rate_percent))

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

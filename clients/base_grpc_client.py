import abc
import time
import clients.base_grpc_client
import distribution
import tensorflow.compat.v1 as tf
import threading
import queue as Queue
import grpc
import json
import numpy as np


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
               error_details, metadata, call_predict_function):
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
    self._call_predict_function = call_predict_function

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
      resp_future = self._call_predict_function(self._stub, self._request,
                                                self._metadata)
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


class BaseGrpcClient(clients.base_client.BaseClient, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def create_grpc_stub(self, grpc_channel):
    raise NotImplementedError()

  @abc.abstractmethod
  def call_predict(self, stub, request, metadata):
    raise NotImplementedError()

  def run(self, requests, num_requests, qps):
    """Runs the benchmark given address, requests and QPS.

      Args:
        requests: Iterable of POST request bodies.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """
    tf.logging.info("Running gRPC benchmark at {} qps".format(qps))

    uri = f"{self._host}:{self._port}"
    tf.logging.info("Inference Server Uri: %s", uri)

    options = [("grpc.max_receive_message_length", 128 * 2**20)]
    grpc_channel = grpc.insecure_channel(
        uri, options=options, compression=self._grpc_compression)
    stub = self.create_grpc_stub(grpc_channel)

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
                      self._grpc_metadata, self.call_predict)
      workers.append(worker)
      worker.start()

      if i % (qps * 10) == 0:
        tf.logging.debug("sent {} requests.".format(i))
      # send requests at a constant rate and adjust for the time it took to send previous request
      pause = interval - (time.time() - previous_worker_start)
      if pause > 0:
        self.sleep(pause)
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

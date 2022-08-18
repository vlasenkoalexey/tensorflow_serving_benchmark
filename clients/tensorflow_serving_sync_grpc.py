import time
import distribution
import threading
import clients.base_client
import clients.tensorflow_serving_grpc
import tensorflow.compat.v1 as tf
import grpc
import json
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class TensorflowServingSyncGrpc(
    clients.tensorflow_serving_grpc.TensorflowServingGrpc):

  def run(self, requests, num_requests, qps):
    """Loadtest the server gRPC endpoint with constant QPS.

      This API is sending gRPC requests in synchronous mode,
      one request per Thread.
      Args:
        requests: Iterable of PredictRequest proto.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """

    tf.logging.info("Running synchronous gRPC benchmark at {} qps".format(qps))

    address = f"{self._host}:{self._port}"
    grpc_channel = grpc.insecure_channel(
        address, compression=self._grpc_compression)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)

    # List appends are thread safe
    success = []
    error = []
    latency = []
    error_details = set()
    dist = distribution.Distribution.factory(self._distribution, qps)

    def _make_grpc_call(i, request):
      """Send GRPC request to Tensorflow Serving endpoint."""
      start_time = time.time()
      try:
        resp = stub.Predict(
            request, self._request_timeout, metadata=self._grpc_metadata)
        success.append(1)
      except Exception as e:
        error.append(1)
        if hasattr(e, "details"):
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
        self.sleep(pause)
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
          "couldn't keep up at current QPS rate, average miss rate:{:.2f}%"
          .format(avg_miss_rate_percent))

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

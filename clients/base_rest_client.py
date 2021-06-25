"""Base abstract REST Client."""

import abc
import json
import time
import threading
import distribution
import clients.base_client
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np


class BaseRestClient(clients.base_client.BaseClient, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_uri(self):
    raise NotImplementedError()

  def run(self, requests, num_requests, qps):
    """Loadtest the server REST endpoint with constant QPS.

      Args:
        requests: Iterable of POST request bodies.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """
    tf.logging.info("Running REST benchmark at %s qps", qps)
    uri = self.get_uri()
    tf.logging.info("Inference Server Uri: %s", uri)

    dist = distribution.Distribution.factory(self._distribution, qps)

    # List appends are thread safe
    success = []
    error = []
    latency = []

    def _make_rest_call(i, request):
      """Send REST POST request to Tensorflow Serving endpoint."""

      headers = self._http_headers
      if len(request) == 2:
        request, request_headers = request
        if request_headers:
          headers = dict(self._http_headers)
          headers.update(request_headers)
      start_time = time.time()
      resp = r.post(uri, data=request, headers=headers)
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

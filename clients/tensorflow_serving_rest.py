import json
import time
import threading
import distribution
import clients.base_client
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np


class TensorflowServingRest(clients.base_client.BaseClient):

  def generate_rest_request_from_tfrecord(self, tfrecord_row):
    """Generate REST inference request's payload."""

    examples = [{"b64": base64.b64encode(row).decode()} for row in tfrecord_row]

    payload = json.dumps({
        "signature_name": self._signature_name,
        "inputs": {
            self._input_name: examples
        },
    })
    return payload

  def get_requests_from_tfrecord(self, path, count, batch_size):
    rows = self.get_tfrecord_rows(path, count, batch_size)
    return [self.generate_rest_request_from_tfrecord(row) for row in rows]

  def generate_rest_request_from_dictionary(self, row_dict):
    payload = json.dumps({
        "signature_name": self._signature_name,
        "inputs": row_dict
    })
    return payload

  def get_requests_from_dictionary(self, path):
    rows = []
    with tf.gfile.GFile(path, "r") as f:
      for line in f:
        row_dict = eval(line)
        rows.append(self.generate_rest_request_from_dictionary(row_dict))
    return rows

  def get_requests_from_file(self, path):
    with tf.gfile.GFile(path, "r") as f:
      j = json.load(f)
      if j is not list:
        j = [j]
      rows = [json.dumps(row) for row in j]
      return rows

  def run(self, requests, num_requests, qps):
    """Loadtest the server REST endpoint with constant QPS.

      Args:
        requests: Iterable of POST request bodies.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """

    tf.logging.info("Running REST benchmark at {} qps".format(qps))

    if self._host.endswith(":predict"):
      uri = _host
    else:
      uri = f"http://{self._host}:{self._port}/v1/models/{self._model_name}:predict"
    dist = distribution.Distribution.factory(self._distribution, qps)

    # List appends are thread safe
    success = []
    error = []
    latency = []

    def _make_rest_call(i, request):
      """Send REST POST request to Tensorflow Serving endpoint."""
      start_time = time.time()
      resp = r.post(uri, data=request, headers=self._http_headers)
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

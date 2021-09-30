"""Client for Google Cloud Vertex AI Prediction.

References:
- https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-custom-models#online_predict_custom_trained-python
"""

import base64
import json
import time
import threading
import distribution
import clients.base_client
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np
from google.cloud.aiplatform import gapic


def generate_request_from_dictionary(row_dict):
    # Vertex AI doesn't support batched requests, therefore it is necessary flatten them    
    batch_size = None
    should_flatten = True
    for key, value in row_dict.items():
        if value and isinstance(value, list) and isinstance(value[0], list):
            if not batch_size:
                batch_size = len(value)
            elif len(value) != batch_size:
                should_flatten = False
                break
        else:
            should_flatten = False
            break

    instances = []
    if should_flatten:
        for i in range(batch_size):
            d = {}
            for key, value in row_dict.items():
                d[key] = value[i]
            instances.append(d)
    else:
        instances.append(row_dict)
    return instances


class VertexGapic(clients.base_client.BaseClient):

  def generate_rest_request_from_tfrecord(self, tfrecord_row):
    raise NotImplementedError()

  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()

  def generate_request_from_dictionary(self, row_dict):
    # Vertex AI doesn't support batched requests, therefore it is necessary flatten them
    batch_size = None
    should_flatten = True
    for key, value in row_dict.items():
        if value and isinstance(value, list) and isinstance(value[0], list):
            if not batch_size:
                batch_size = len(value)
            elif len(value) != batch_size:
                should_flatten = False
                break
        else:
            should_flatten = False
            break
            
    instances = []
    if should_flatten:
        for i in range(batch_size):
            d = {}
            for key, value in row_dict.items():
                d[key] = value[i]
            instances.append(d)
    else:
        instances.append(row_dict)
    return instances

  def get_requests_from_dictionary(self, path):
    rows = []
    with tf.gfile.GFile(path, "r") as f:
      for line in f:
        row_dict = json.loads(line)
        rows.append(generate_request_from_dictionary(row_dict))
    return rows

  def get_requests_from_file(self, path):
    raise NotImplementedError()

  def run(self, requests, num_requests, qps):
    """Loadtest the server REST endpoint with constant QPS.

      Args:
        requests: Iterable of POST request bodies.
        num_requests: Number of requests.
        qps: The number of requests being sent per second.
    """
    
    predict_client_options = {"api_endpoint": self._host}
    client = gapic.PredictionServiceClient(client_options=predict_client_options)
    
    tf.logging.info("Running Vertex GAPIC benchmark at %s qps", qps)
    tf.logging.info("Inference Server Uri: %s", self._host)

    dist = distribution.Distribution.factory(self._distribution, qps)

    # List appends are thread safe
    success = []
    error = []
    latency = []

    # TODO: move this functionality to base class and expose _make_call for overwride.
    def _make_call(i, request):
      """Send request to Vertex AI."""

      start_time = time.time()
      try:
        _ = client.predict(endpoint=self._model_name, instances=request)  
        success.append(1)
      except Error as er:
        tf.logging.error(resp.json())
        error.append(1)
        
      latency.append(time.time() - start_time)
      if len(latency) % (10 * qps) == 0:
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
      thread = threading.Thread(target=_make_call, args=(i, request))
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
        

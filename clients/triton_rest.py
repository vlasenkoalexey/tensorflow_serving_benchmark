"""Client for Triton Inference Server using REST API.

References:
- https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md
- https://github.com/triton-inference-server/client/tree/master/src/python/examples
- https://github.com/triton-inference-server/client/blob/master/src/python/library/tritonclient/http/__init__.py
"""

import json
import time
import threading
import distribution
import clients.base_client
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils as triton_utils
from tensorflow.python.framework import dtypes


class TritonRest(clients.base_client.BaseClient):
  
  def generate_rest_request_from_dictionary(self, row_dict):
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
      triton_request_input = httpclient.InferInput(
        key, 
        list(numpy_value.shape), 
        triton_utils.np_to_triton_dtype(t))
      triton_request_input.set_data_from_numpy(numpy_value, binary_data=True) # binary_data=True by default
      triton_request_inputs.append(triton_request_input)
    # https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/http/__init__.py#L81
    # Returns tuple - request and request len to pass in Infer-Header-Content-Length header
    (request, json_size) = httpclient._get_inference_request(
      inputs=triton_request_inputs, 
      request_id="", 
      outputs=None,
      sequence_id=0,
      sequence_start=0,
      sequence_end=0,
      priority=0, 
      timeout=None)
    
    headers = {}
    if json_size:
      headers["Inference-Header-Content-Length"] = str(json_size)
    return (request, headers)
    
    
  def get_requests_from_dictionary(self, path):
    rows = []
    with tf.gfile.GFile(path, "r") as f:
      for line in f:
        row_dict = eval(line)
        rows.append(self.generate_rest_request_from_dictionary(row_dict))
    return rows
  
  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()

  def get_requests_from_file(self, path):
    raise NotImplementedError()
    
  def get_uri(self):
    if self._host.startswith("http"):
      return self._host
    else:
      # https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#httprest
      if self._model_version:
        return f"http://{self._host}:{self._port}/v2/models/{self._model_name}/versions/{self._model_version}/infer"
      else:
        return f"http://{self._host}:{self._port}/v2/models/{self._model_name}/infer"    

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
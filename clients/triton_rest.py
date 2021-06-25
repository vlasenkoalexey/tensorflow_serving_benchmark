"""Client for Triton Inference Server using REST API.

References:
-
https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#httprest
-
https://github.com/triton-inference-server/client/tree/master/src/python/examples
-
https://github.com/triton-inference-server/client/blob/master/src/python/library/tritonclient/http/__init__.py
"""

import json
import time
import threading
import distribution
import clients.base_rest_client
import clients.utils
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np
import tritonclient.http as triton_httpclient
import tritonclient.utils as triton_utils
from tensorflow.python.framework import dtypes


class TritonRest(clients.base_rest_client.BaseRestClient):

  def generate_rest_request_from_dictionary(self, row_dict):
    triton_request_inputs = []
    for key, value in row_dict.items():
      t = clients.utils.get_type(value, self._default_float_type,
                                 self._default_int_type)
      if t == np.object_:
        value = clients.utils.map_multi_dimensional_list(
            value, lambda s: s.encode("utf-8"))
      numpy_value = np.array(value, dtype=t)
      triton_request_input = triton_httpclient.InferInput(
          key, list(numpy_value.shape), triton_utils.np_to_triton_dtype(t))
      triton_request_input.set_data_from_numpy(
          numpy_value, binary_data=True)  # binary_data=True by default
      triton_request_inputs.append(triton_request_input)
    # https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/http/__init__.py#L81
    # Returns tuple - request and request len to pass in Infer-Header-Content-Length header
    (request, json_size) = triton_httpclient._get_inference_request(
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

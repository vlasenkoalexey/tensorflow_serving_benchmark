"""Client for TensorFlow Serving Inference Server using REST API.

References:
- https://www.tensorflow.org/tfx/serving/api_rest
- https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
- https://github.com/tensorflow/serving
"""

import json
import time
import threading
import distribution
import clients.base_rest_client
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np


class TensorflowServingRest(clients.base_rest_client.BaseRestClient):

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

  def get_uri(self):
    if self._host.startswith("http"):
      return self._host
    else:
      # https://www.tensorflow.org/tfx/serving/api_rest
      if self._model_version:
        return f"http://{self._host}:{self._port}/v1/models/{self._model_name}/versions/{self._model_version}:predict"
      else:
        return f"http://{self._host}:{self._port}/v1/models/{self._model_name}:predict"

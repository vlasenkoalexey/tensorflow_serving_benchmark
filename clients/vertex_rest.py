"""Client for Vertex AI using REST API.

References:
- https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-custom-models#online_predict_custom_trained-drest
"""

import base64
import json
import time
import threading
import distribution
import clients.base_rest_client
from clients.vertex_gapic import generate_request_from_dictionary
import tensorflow.compat.v1 as tf
import requests as r
import numpy as np


class VertexRest(clients.base_rest_client.BaseRestClient):

  def generate_rest_request_from_tfrecord(self, tfrecord_row):
    raise NotImplementedError()

  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()
    
  def get_requests_from_file(self, path):
    raise NotImplementedError()    

  def get_requests_from_dictionary(self, path):
    rows = []
    with tf.gfile.GFile(path, "r") as f:
      for line in f:
        row_dict = json.loads(line)
        rows.append(
          json.dumps({
            "instances": generate_request_from_dictionary(row_dict) 
          }))
    return rows

  def get_uri(self):
    return self._host

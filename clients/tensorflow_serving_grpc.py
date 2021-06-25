"""Client for benchmarking TensorFlow Serving Inference Server using gRPC API.

References:
- https://github.com/tensorflow/serving
- https://github.com/tensorflow/serving/tree/master/tensorflow_serving/apis
"""

import time
import clients.base_grpc_client
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
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class TensorflowServingGrpc(clients.base_grpc_client.BaseGrpcClient):

  def generate_grpc_request_from_tfrecord(self, tfrecord_row):
    """Generate gRPC inference request with payload."""

    request = predict_pb2.PredictRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = self._signature_name
    request.inputs[self._input_name].CopyFrom(
        tf.make_tensor_proto(tfrecord_row, dtype=tf.string))
    return request

  def get_requests_from_tfrecord(self, path, count, batch_size):
    rows = self.get_tfrecord_rows(path, count, batch_size)
    return [self.generate_grpc_request_from_tfrecord(row) for row in rows]

  def generate_grpc_request_from_dictionary(self, row_dict):
    """Generate gRPC inference request with payload."""

    def isSubType(value, t):
      while isinstance(value, list):
        if len(value) > 0:
          value = value[0]
        else:
          return False
      return isinstance(value, t)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = self._signature_name
    for key, value in row_dict.items():
      proto = None
      if self._default_int_type and (isSubType(value, int) or
                                     "/values" in key or "/indices" in key):
        proto = tf.make_tensor_proto(value, dtype=self._default_int_type)
      elif self._default_float_type and isSubType(value, float):
        proto = tf.make_tensor_proto(value, dtype=self._default_float_type)
      else:
        proto = tf.make_tensor_proto(value)
      request.inputs[key].CopyFrom(proto)
    return request

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
          ProtoParseJson(json.dumps(row), predict_pb2.PredictRequest())
          for row in j
      ]
      return rows

  def create_grpc_stub(self, grpc_channel):
    return prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)

  def call_predict(self, stub, request, metadata):
    return stub.Predict.future(
        request, timeout=self._request_timeout, metadata=metadata)

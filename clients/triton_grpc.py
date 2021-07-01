"""Client for benchmarking Triton Inference Server using gRPC API.

References:
-
https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#grpc
-
https://github.com/triton-inference-server/client/tree/master/src/python/examples
-
https://github.com/triton-inference-server/client/blob/master/src/python/library/tritonclient/grpc/__init__.py
"""

import time
import clients.base_grpc_client
import clients.utils
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
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tensorflow.python.framework import dtypes

import tritonclient.grpc as triton_grpcclient
import tritonclient.utils as triton_utils


class TritonGrpc(clients.base_grpc_client.BaseGrpcClient):

  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()

  def generate_grpc_request_from_dictionary(self, row_dict):
    triton_request_inputs = []
    for key, value in row_dict.items():
      t = clients.utils.get_type(key, value, self._default_float_type,
                                 self._default_int_type)
      if t == np.object_:
        value = clients.utils.map_multi_dimensional_list(
            value, lambda s: s.encode("utf-8"))
      numpy_value = np.array(value, dtype=t)
      triton_request_input = triton_grpcclient.InferInput(
          key, list(numpy_value.shape), triton_utils.np_to_triton_dtype(t))
      triton_request_input.set_data_from_numpy(numpy_value)
      triton_request_inputs.append(triton_request_input)
    # https://github.com/triton-inference-server/client/blob/530bcac5f1574aa2222930076200544eb274245c/src/python/library/tritonclient/grpc/__init__.py#L64
    return triton_grpcclient._get_inference_request(
        model_name=self._model_name,
        inputs=triton_request_inputs,
        model_version=self._model_version,
        request_id="",
        outputs=None,
        sequence_id=0,
        sequence_start=0,
        sequence_end=0,
        priority=0,
        timeout=None)

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
          ProtoParseJson(json.dumps(row), service_pb2.ModelInferRequest())
          for row in j
      ]
      return rows

  def create_grpc_stub(self, grpc_channel):
    return service_pb2_grpc.GRPCInferenceServiceStub(grpc_channel)

  def call_predict(self, stub, request, metadata):
    return stub.ModelInfer.future(
        request, timeout=self._request_timeout, metadata=metadata)

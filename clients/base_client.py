"""Abstract base class for benchmark clients."""

import abc
import base64
import time
from enum import Enum
import tensorflow.compat.v1 as tf


class RequestFormat(Enum):
  FILE = 1
  TFRECORD = 2
  DICTIONARY = 3


class BaseClient(metaclass=abc.ABCMeta):

  def __init__(self, host, port, model_name, model_version, signature_name,
               distribution, input_name, default_int_type, default_float_type,
               http_headers, grpc_metadata, grpc_compression, request_timeout,
               busy_sleep):
    self._host = host
    self._port = port
    self._model_name = model_name
    self._model_version = model_version
    self._signature_name = signature_name
    self._distribution = distribution
    self._input_name = input_name
    self._default_int_type = default_int_type
    self._default_float_type = default_float_type
    self._http_headers = http_headers
    self._grpc_metadata = grpc_metadata
    self._grpc_compression = grpc_compression
    self._request_timeout = request_timeout
    self._busy_sleep = busy_sleep

  @abc.abstractmethod
  def get_requests_from_tfrecord(self, path, count, batch_size):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_requests_from_dictionary(self, path):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_requests_from_file(self, path):
    raise NotImplementedError()

  def get_tfrecord_rows(self, tfrecord_dataset_path, count, batch_size):
    tf.logging.info("loading data for prediction")
    if not count:
      count = 1
    compression_type = "GZIP" if tfrecord_dataset_path.endswith(".gz") else None
    dataset = tf.data.TFRecordDataset(
        tf.gfile.Glob(tfrecord_dataset_path), compression_type=compression_type)
    if batch_size is not None:
      dataset = dataset.batch(batch_size)
    rows = []
    itr = tf.data.make_initializable_iterator(dataset.repeat())
    next = itr.get_next()
    with tf.Session() as sess:
      sess.run(itr.initializer)
      for _ in range(count):
        inputs = sess.run(next)
        rows.append(inputs)
    return rows

  def sleep(self, seconds: float) -> None:
    if seconds < 0:
      raise ValueError("Sleep seconds can't be negative.", seconds)
    if self._busy_sleep:
      wakeup = time.time() + seconds
      while time.time() < wakeup:
        time.sleep(0.0)
    else:
      time.sleep(seconds)

  def get_requests(self, format, path, count, batch_size):
    if format == RequestFormat.FILE:
      return self.get_requests_from_file(path)
    elif format == RequestFormat.TFRECORD:
      return self.get_requests_from_tfrecord(path, count, batch_size)
    elif format == RequestFormat.DICTIONARY:
      return self.get_requests_from_dictionary(path)
    else:
      raise ValueError("invalid request format")

  @abc.abstractmethod
  def run(requests, num_requests, qps):
    """Runs the benchmark given address, requests and QPS.

    Args:
      requests: Iterable of POST request bodies.
      num_requests: Number of requests.
      qps: The number of requests being sent per second.
    """
    raise NotImplementedError()

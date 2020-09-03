"""A loadtest script which sends request via GRPC to TF inference server.
Adapted from:
https://github.com/tensorflow/tpu/blob/master/models/experimental/inference/load_test_client.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io
import json
import threading
import time
import grpc

import numpy as np
from PIL import Image
import queue as Queue
import requests as r
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import distribution

tf.app.flags.DEFINE_integer('num_requests', 20, 'Total # of requests sent.')
tf.app.flags.DEFINE_string('qps_range', '', 'Desired client side request QPS in'
                           'one of the following formats:'
                           'qps - benchmark at one QPS'
                           'start, stop - benchmark at QPS range [start, stop)'
                           'start, stop, step - benchmark at QPS range [start, stop) with step'
                           '[qps1, qps2] - benchmark at give QPS range values')
tf.app.flags.DEFINE_float('request_timeout', 300.0,
                          'Timeout for inference request.')
tf.app.flags.DEFINE_string('model_name', '',
                           'Name of the model being served on the ModelServer')
tf.app.flags.DEFINE_string('signature_name', 'serving_default',
                           'Name of the model signature on the ModelServer')
tf.app.flags.DEFINE_string("host", "localhost",
                    "Host name to connect to, localhost by default.")
tf.app.flags.DEFINE_integer("port", None, "Port to connect to.")
tf.app.flags.DEFINE_enum('mode', 'grpc', ['grpc', 'sync_grpc', 'rest'],
                         'Benchmark mode: gRPC, synchronous gRPC, or REST')
tf.app.flags.DEFINE_enum('distribution', 'uniform', ['uniform', 'poisson', 'pareto'],
                         'Distribution')
tf.app.flags.DEFINE_string('tfrecord_dataset_path', '', 'The path to data.')
tf.app.flags.DEFINE_string('input_name', 'input',
                           'The name of the model input tensor.')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Per request batch size.')
tf.app.flags.DEFINE_string(
    'api_key', '',
    'API Key for ESP service if authenticating external requests.')
tf.app.flags.DEFINE_string('csv_report_filename', '', 'Filename to generate report')


FLAGS = tf.app.flags.FLAGS


class Worker(object):
  """A loadtest worker which sends RPC request."""

  __slot__ = ('_id', '_request', '_stub', '_queue', '_success', '_start_time',
              '_end_time', '_qps', '_num_requests', '_metadata')

  def __init__(self, index, request, stub, queue, qps, num_requests, metadata):
    self._id = index
    self._request = request
    self._stub = stub
    self._queue = queue
    self._qps = qps
    self._num_requests = num_requests
    self._success = None
    self._start_time = None
    self._end_time = None
    self._metadata = metadata

  def start(self):
    """Start to send request."""

    def _callback(resp_future):
      """Callback for aynchronous inference request sent."""
      exception = resp_future.exception()
      if exception:
        self._success = False
        tf.logging.error(exception)
      else:
        self._success = True
      self._end_time = time.time()
      self._queue.get()
      self._queue.task_done()
      processed_count = self._num_requests - self._queue.qsize()
      if processed_count % (10 * self._qps) == 0:
        tf.logging.info('received {} responses'.format(processed_count))

    def _send_rpc():
      self._start_time = time.time()
      resp_future = self._stub.Predict.future(self._request,
                                              FLAGS.request_timeout,
                                              metadata=self._metadata)
      resp_future.add_done_callback(_callback)

    _send_rpc()

  def cancel(self):
    self._rpc.StartCancel()

  @property
  def success_count(self):
    return int(self._success)

  @property
  def error_count(self):
    return int(not self._success)

  @property
  def latency(self):
    if not (self._start_time and self._end_time):
      raise Exception('Request is not complete yet.')
    return self._end_time - self._start_time


def run_grpc_load_test(requests, qps, stub):
  """Loadtest the server gRPC endpoint with constant QPS.
  Args:
    requests: List of PredictRequest proto.
    qps: The number of requests being sent per second.
    stub: The model server stub to which send inference requests.
  """

  tf.logging.info('Running gRPC benchmark at {} qps'.format(qps))
  dist = distribution.Distribution.factory(FLAGS.distribution, qps)
  num_requests = len(requests)
  metadata = []
  if FLAGS.api_key:
    metadata.append(('x-api-key', FLAGS.api_key))

  q = Queue.Queue()
  intervals = []
  for i in range(num_requests):
    q.put(i)
    intervals.append(dist.next())

  workers = []
  miss_rate_percent = []
  start = time.time()
  previous_worker_start = start
  for i in range(num_requests):
    request = requests[i]
    interval = intervals[i]
    worker = Worker(i, request, stub, q, qps, num_requests, metadata)
    workers.append(worker)
    worker.start()
    if i % (qps * 10) == 0:
      tf.logging.info('sent {} requests.'.format(i))
    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = 100 * ((time.time() - previous_worker_start) - interval) / (interval)
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()

  # block until all workers are done
  q.join()
  acc_time = time.time() - start
  success_count = 0
  error_count = 0
  latency = []
  worker_end_time = start
  for w in workers:
    success_count += w.success_count
    error_count += w.error_count
    latency.append(w.latency)
    worker_end_time = w._end_time if w._end_time > worker_end_time else worker_end_time

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn('couldn\'t keep up at current QPS rate, average miss rate:{:.2f}%'.format(avg_miss_rate_percent))

  tf.logging.info('num_qps:{:.2f} requests/second: {:.2f} #success:{} #error:{} '
                  'latencies: [avg:{:.2f}ms p50:{:.2f}ms p90:{:.2f}ms p99:{:.2f}ms]'.format(
                      qps, num_requests / acc_time, success_count, error_count,
                      np.average(latency) * 1000, np.percentile(latency, 50) * 1000,
                      np.percentile(latency, 90) * 1000, np.percentile(latency, 99) * 1000))

  return {
    'reqested_qps': qps,
    'actual_qps': num_requests / acc_time,
    'success': success_count,
    'error': error_count,
    'time': acc_time,
    'avg_latency': np.average(latency) * 1000,
    'p50': np.percentile(latency, 50) * 1000,
    'p90': np.percentile(latency, 90) * 1000,
    'p99': np.percentile(latency, 99) * 1000,
    'avg_miss_rate_percent': avg_miss_rate_percent
  }


def run_synchronous_grpc_load_test(requests, qps, stub):
  """Loadtest the server gRPC endpoint with constant QPS.

  This API is sending gRPC requests in synchronous mode,
  one request per Thread.
  Args:
    requests: List of PredictRequest proto.
    qps: The number of requests being sent per second.
    stub: The model server stub to which send inference requests.
  """

  tf.logging.info('Running gRPC benchmark at {} qps'.format(qps))
  # List appends are thread safe
  num_requests = len(requests)
  success = []
  error = []
  latency = []
  metadata = []
  if FLAGS.api_key:
    metadata.append(('x-api-key', FLAGS.api_key))
  dist = distribution.Distribution.factory(FLAGS.distribution, qps)

  def _make_grpc_call(i):
    """Send GRPC POST request to Tensorflow Serving endpoint."""
    start_time = time.time()
    try:
      resp = stub.Predict(requests[i],
                      FLAGS.request_timeout,
                      metadata=metadata)
      success.append(1)
    except Exception as e:
      print(e)
      error.append(1)

    latency.append(time.time() - start_time)
    if len(latency) % (qps * 10) == 0:
      tf.logging.info('received {} responses.'.format(len(latency)))

  intervals = []
  for i in range(num_requests):
    intervals.append(dist.next())

  thread_lst = []
  miss_rate_percent = []
  start_time = time.time()
  previous_worker_start = start_time
  for i in range(num_requests):
    interval = intervals[i]
    thread = threading.Thread(target=_make_grpc_call, args=(i,))
    thread_lst.append(thread)
    thread.start()
    if i % (qps * 10) == 0:
      tf.logging.info('sent {} requests.'.format(i))

    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = 100 * ((time.time() - previous_worker_start) - interval) / (interval)
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()

  for thread in thread_lst:
    thread.join()

  acc_time = time.time() - start_time

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn('couldn\'t keep up at current QPS rate, average miss rate:{:.2f}%'.format(avg_miss_rate_percent))

  tf.logging.info('num_qps:{} requests/second: {:.2f} #success:{} #error:{} '
                  'latencies: [avg:{:.2f}ms p50:{:.2f}ms p90:{:.2f}ms p99:{:.2f}ms]'.format(
                      qps, num_requests / acc_time, sum(success), sum(error),
                      np.average(latency) * 1000, np.percentile(latency, 50) * 1000,
                      np.percentile(latency, 90) * 1000, np.percentile(latency, 99) * 1000))
  return {
    'reqested_qps': qps,
    'actual_qps': num_requests / acc_time,
    'success': sum(success),
    'error': sum(error),
    'time': acc_time,
    'avg_latency': np.average(latency) * 1000,
    'p50': np.percentile(latency, 50) * 1000,
    'p90': np.percentile(latency, 90) * 1000,
    'p99': np.percentile(latency, 99) * 1000,
    'avg_miss_rate_percent': avg_miss_rate_percent
  }


def run_rest_load_test(requests, qps, address):
  """Run inference load test against REST endpoint."""

  tf.logging.info('Running REST benchmark at {} qps'.format(qps))

  address = 'http://{}/v1/models/{}:predict'.format(address, FLAGS.model_name)
  dist = distribution.Distribution.factory(FLAGS.distribution, qps)

  # List appends are thread safe
  num_requests = len(requests)
  success = []
  error = []
  latency = []

  def _make_rest_call(i):
    """Send REST POST request to Tensorflow Serving endpoint."""
    start_time = time.time()
    resp = r.post(address, data=requests[i])
    latency.append(time.time() - start_time)
    if len(latency) % (10 * qps) == 0:
      tf.logging.info('received {} responses.'.format(len(latency)))
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
  for i in range(num_requests):
    interval = intervals[i]
    thread = threading.Thread(target=_make_rest_call, args=(i,))
    thread_lst.append(thread)
    thread.start()
    if i % (10 * qps) == 0:
      tf.logging.info('sent {} requests.'.format(i))

    # send requests at a constant rate and adjust for the time it took to send previous request
    pause = interval - (time.time() - previous_worker_start)
    if pause > 0:
      time.sleep(pause)
    else:
      missed_delay = 100 * ((time.time() - previous_worker_start) - interval) / (interval)
      miss_rate_percent.append(missed_delay)
    previous_worker_start = time.time()

  for thread in thread_lst:
    thread.join()

  acc_time = time.time() - start_time

  avg_miss_rate_percent = 0
  if len(miss_rate_percent) > 0:
    avg_miss_rate_percent = np.average(miss_rate_percent)
    tf.logging.warn('couldn\'t keep up at current QPS rate, average miss rate:{:.2f}%'.format(avg_miss_rate_percent))

  tf.logging.info('num_qps:{} requests/second: {} #success:{} #error:{} '
                  'latencies: [avg:{:.2f}ms p50:{:.2f}ms p90:{:.2f}ms p99:{:.2f}ms]'.format(
                      qps, num_requests / acc_time, sum(success), sum(error),
                      np.average(latency) * 1000, np.percentile(latency, 50) * 1000,
                      np.percentile(latency, 90) * 1000, np.percentile(latency, 99) * 1000))
  return {
    'reqested_qps': qps,
    'actual_qps': num_requests / acc_time,
    'success': sum(success),
    'error': sum(error),
    'time': acc_time,
    'avg_latency': np.average(latency) * 1000,
    'p50': np.percentile(latency, 50) * 1000,
    'p90': np.percentile(latency, 90) * 1000,
    'p99': np.percentile(latency, 99) * 1000,
    'avg_miss_rate_percent': avg_miss_rate_percent
  }

def get_rows():
  tf.logging.info("loading data for prediction")
  dataset = tf.data.TFRecordDataset(gfile.Glob(FLAGS.tfrecord_dataset_path))
  if FLAGS.batch_size is not None:
      dataset = dataset.batch(FLAGS.batch_size)
  rows = []
  itr = tf.data.make_initializable_iterator(dataset.repeat())
  next = itr.get_next()
  with tf.Session() as sess:
      sess.run(itr.initializer)
      for _ in range(FLAGS.num_requests):
          inputs = sess.run(next)
          rows.append(inputs)
  return rows


def generate_rest_request(tfrecord_row):
  """Generate REST inference request's payload."""

  examples = [{ 'b64': base64.b64encode(row).decode() } for row in tfrecord_row]

  payload = json.dumps({
      'signature_name': FLAGS.signature_name,
      'inputs': {
        FLAGS.input_name: examples
      },
  })
  return payload

def generate_grpc_request(tfrecord_row):
  """Generate gRPC inference request with payload."""

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name
  request.inputs[FLAGS.input_name].CopyFrom(tf.make_tensor_proto(tfrecord_row, dtype=tf.string))
  return request


def get_qps_range(qps_range_string):
  qps_range_string = qps_range_string.strip()
  if qps_range_string.startswith('[') and qps_range_string.endswith(']'):
    qps_range_string = qps_range_string.lstrip('[').rstrip(']')
    qps_range_list = list(map(lambda v: int(v), qps_range_string.split(',')))
    return qps_range_list

  qps_range_list = list(map(lambda v: int(v), qps_range_string.split(',')))
  qps_range_start = 0
  qps_range_step = 1
  if (len(qps_range_list) == 1):
    return [qps_range_list[0]]
  elif (len(qps_range_list) == 2):
    return range(qps_range_list[0], qps_range_list[1])
  elif (len(qps_range_list) == 3):
    return range(qps_range_list[0], qps_range_list[1], qps_range_list[2])
  else:
    raise ValueError('Invalid argument qps_range:' + qps_range_start)

def merge_results(results, result):
  for key, value in result.items():
    if key not in results:
      results[key] = [value]
    else:
      results[key].append(value)

def main(argv):
  del argv
  tf.disable_v2_behavior()

  if FLAGS.qps_range is None or FLAGS.qps_range =='':
    tf.logging.error('Please specify qps_range')
    exit(1)

  address = "{}:{}".format(FLAGS.host, FLAGS.port)
  tf.logging.info('ModelServer at: {}'.format(address))

  tf.logging.info('Loading data')
  rows = get_rows()

  results = {}

  if FLAGS.mode == 'grpc' or FLAGS.mode == 'sync_grpc':
    grpc_requests = [generate_grpc_request(row) for row in rows]
    grpc_channel = grpc.insecure_channel(address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)
    for qps in get_qps_range(FLAGS.qps_range):
      result = run_grpc_load_test(grpc_requests, qps, stub) \
        if FLAGS.mode == 'grpc' else run_synchronous_grpc_load_test(grpc_requests, qps, stub)
      merge_results(results, result)
  elif FLAGS.mode == 'rest':
    rest_requests = [generate_rest_request(row) for row in rows]
    for qps in get_qps_range(FLAGS.qps_range):
      result = run_rest_load_test(rest_requests, qps, address)
      merge_results(results, result)
  else:
    raise ValueError('Invalid --mode:' + FLAGS.mode)

  if FLAGS.csv_report_filename is not None:
    import pandas as pd
    pd.DataFrame.from_dict(results).to_csv(FLAGS.csv_report_filename)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.plot('actual_qps', 'p50', data=results, label='p50')
    plt.plot('actual_qps', 'p90', data=results, label='p90')
    plt.plot('actual_qps', 'p99', data=results, label='p99')
    plt.plot('actual_qps', 'avg_latency', data=results, label='avg_latency')
    plt.legend()
    plt.savefig(FLAGS.csv_report_filename + '.png')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
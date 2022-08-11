"""Benchmark runner tool to start inference docker container and corresponding benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from requests.sessions import default_headers

import os
import queue as Queue
import requests
import threading
import multiprocessing
import time
import numpy as np
import os
import subprocess
import io

from threading import Thread
from queue import Queue

# Disable GPU, so tensorflow initializes faster and doesn't consume GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf

import benchmark


tf.app.flags.DEFINE_string("server_command", "",
                           "Command to start Model Server")
tf.app.flags.DEFINE_string("health_check_uri", "",
                           "Uri to check that Model Server is stared")
tf.app.flags.DEFINE_integer("startup_timeout_seconds", 180,
                            "Timeout for Model Server startup")
tf.app.flags.DEFINE_boolean("keep_server_alive", False,
                            "Whether to keep Model Server alive after benchmark is completed")

FLAGS = tf.app.flags.FLAGS

_ENABLE_MODEL_SERVER_LOGGING = True

def _reader(pipe, pid, pipe_name: str):
  """Redirect pipe to logging.

  Redirects pipe to log level warning if pipe name is stderr, redicts to log
  level info otherwise.

  Args:
    pipe: either stdout or stderr.
    pid: pid of source.
    pipe_name: either "stdout" or "stderr".
  """
  if not pipe or not _ENABLE_MODEL_SERVER_LOGGING:
    return

  try:
    with pipe:
      for line in iter(pipe.readline, ""):
        if _ENABLE_MODEL_SERVER_LOGGING:
          if pipe_name == "stderr":
            tf.logging.warning(line.strip().strip('\n'))
          else:
            tf.logging.info(line.strip().strip('\n'))
        else:
            tf.logging.info("pipe_name %s exited", pipe_name)
            return
  finally:
    tf.logging.info("exited process %s", pid)

def _run_process(command, **kwargs):
    global _ENABLE_MODEL_SERVER_LOGGING
    args = [arg.strip(' ').strip('\n') for arg in command.split(' ') if arg.strip(' ')]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        encoding='utf-8',
        start_new_session=True
    )

    q = Queue()
    _ENABLE_MODEL_SERVER_LOGGING = True
    Thread(target=_reader, args=[proc.stdout, q, 'stdout']).start()
    Thread(target=_reader, args=[proc.stderr, q, 'stderr']).start()
    return proc


def _stop_model_server_logging():
    global _ENABLE_MODEL_SERVER_LOGGING
    _ENABLE_MODEL_SERVER_LOGGING = False


def _stop_model_server_process(process):
  global _ENABLE_MODEL_SERVER_LOGGING
  _ENABLE_MODEL_SERVER_LOGGING = False
  try:
    process.terminate()
  except:
    pass


def _wait_for_model_server(health_check_uri, startup_timeout_seconds):
    timeout = startup_timeout_seconds
    while timeout > 0:
        try:
            response = requests.get(health_check_uri, timeout=1)
            if response.status_code == 200:
                print("Model Server is up and running")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)

        print (f"Waiting for container, {timeout} seconds remaining until timeout")
        timeout -= 1

    if timeout == 0:
        return False


def main(argv):
  if FLAGS.server_command:
    tf.logging.info("Starting Model Server")
    tf.logging.info(FLAGS.server_command)
    proc = _run_process(FLAGS.server_command)
    tf.logging.info("Started Model Server with command process id: %s", proc.pid)
    if not _wait_for_model_server(FLAGS.health_check_uri, FLAGS.startup_timeout_seconds):
        tf.logging.info("Model Server failed to initialize")
        tf.logging.info("Stopping Model Server")
        _stop_model_server_process(proc)
        tf.logging.info("Stopped Model Server")
        exit(1)
  else:
    if not _wait_for_model_server(FLAGS.health_check_uri, FLAGS.startup_timeout_seconds):
        tf.logging.info("Model Server health endpoint didn't repsond")
        exit(1)

  tf.logging.info("Starting benchmark")
  tf.logging.info("----------------------------------------------------------")
  benchmark.main(argv)
  if FLAGS.server_command:
      if not FLAGS.keep_server_alive:
        tf.logging.info("Stopping Model Server")
        _stop_model_server_process(proc)
        tf.logging.info("Stopped Model Server")
      else:
        _stop_model_server_logging()
        tf.logging.info("Keeping Model Server alive")
  tf.logging.info("exiting")
  exit(0)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  logger = tf.get_logger()
  logger.propagate = False
  tf.app.run(main)
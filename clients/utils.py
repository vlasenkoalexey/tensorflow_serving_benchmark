import numpy as np
from tensorflow.python.framework import dtypes


def is_subtype(value, t):
  while isinstance(value, list):
    if len(value) > 0:
      value = value[0]
    else:
      return False
  return isinstance(value, t)


def get_type(value, default_float_type, default_int_type):
  if is_subtype(value, float):
    return dtypes.as_dtype(
        default_float_type
    ).as_numpy_dtype if default_float_type else numpy.float32
  elif is_subtype(value, int):
    return dtypes.as_dtype(
        default_int_type).as_numpy_dtype if default_int_type else numpy.int32
  elif is_subtype(value, str):
    return np.object_
  elif is_subtype(value, bool):
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

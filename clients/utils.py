import numpy as np
from tensorflow.python.framework import dtypes


def is_subtype(value, t):
  while isinstance(value, list):
    if len(value) > 0:
      value = value[0]
    else:
      return False
  return isinstance(value, t)


def get_type(key, value, default_float_type, default_int_type):
  if is_subtype(value, float):
    return dtypes.as_dtype(
        default_float_type).as_numpy_dtype if default_float_type else np.float32
  elif is_subtype(value, str):
    return np.object_
  elif is_subtype(value, bool):
    return np.bool
  elif is_subtype(value, int) or "/values" in key or "/indices" in key:
    return dtypes.as_dtype(
        default_int_type).as_numpy_dtype if default_int_type else np.int32
  else:
    raise ValueError(f"Can't detect type for key {key} value {value}")


def map_multi_dimensional_list(l, transform):
  if type(l) == list and len(l) > 0:
    if type(l[0]) != list:
      return [transform(v) for v in l]
    else:
      return [map_multi_dimensional_list(v, transform) for v in l]
  else:
    return []

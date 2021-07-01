"""Graph generator for CSV reports produced by benchmark tool."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.filters import gaussian_filter1d


def parse_report_csv(file_name):
  with open(file_name, newline='') as f:
    reader = csv.reader(f)

    d = {}
    index_to_key = {}
    for row in reader:
      if not d:
        for index in range(len(row)):
          key = row[index]
          index_to_key[index] = key
          d[key] = []
      else:
        for index in range(len(row)):
          if index_to_key[index] != 'title':
            d[index_to_key[index]].append(float(row[index]))
          else:
            d[index_to_key[index]].append(row[index])
  return d


def generate_graph(filename_legend_pairs,
                   title='Average Latency',
                   column='avg_latency',
                   min_x=0,
                   max_x=None,
                   min_y=0,
                   max_y=None,
                   sigma=0.1):
  """Generate graph for the given CSV files.

    Args:
      filename_legend_pairs: CSV file path, or a pair of CSV file path and line
        legend title on the report.
      title: graph title.
      column: column in CSV file to take Y values from.
      min_x: value to overwrite minimum X on the graph.
      max_x: value to overwrite maxiumum X on the graph.
      min_y: value to overwrite minimum Y on the graph.
      min_y: value to overwrite minimum Y on the graph.
      sigma: smoothing factor, the bigger value, the smother the graph.

    Returns:
      pyplot graph.
    """
  fig, ax = plt.subplots(facecolor=(1, 1, 1))
  ax.set_title(title)
  ax.set_xlabel('QPS')
  ax.set_ylabel('Latency(ms)')

  for pair in filename_legend_pairs:
    if len(pair) == 2:
      file_name, legend = pair
      r = parse_report_csv(file_name)
    else:
      file_name = pair
      r = parse_report_csv(file_name)
      if 'title' in r and len(r['title']) > 0 and r['title'][0]:
        legend = r['title'][0]
      else:
        legend = file_name

    x = np.array(r['actual_qps'])
    y = r[column]
    if max_y:
      y = np.array(np.clip(r[column], min_y, max_y))

    y_smoothed = gaussian_filter1d(y, sigma=sigma)
    ax.plot(x, y_smoothed, label=legend)
    ax.legend()

  _min_x, _max_x, _min_y, _max_y = ax.axis()
  if not max_x:
    max_x = _max_x
  if not max_y:
    max_y = _max_y
  ax.axis((min_x, max_x, min_y, max_y))
  return plt


def generate_avg_latency_graph(filename_legend_pairs,
                               title='Average Latency',
                               min_x=0,
                               max_x=None,
                               min_y=0,
                               max_y=None,
                               sigma=0.1):
  """Generate average latency graph for the given CSV files.

    Args:
      filename_legend_pairs: CSV file path, or a pair of CSV file path and line
        legend title on the report.
      title: graph title.
      min_x: value to overwrite minimum X on the graph.
      max_x: value to overwrite maxiumum X on the graph.
      min_y: value to overwrite minimum Y on the graph.
      min_y: value to overwrite minimum Y on the graph.
      sigma: smoothing factor, the bigger value, the smother the graph.

    Returns:
      pyplot graph.
    """
  return generate_graph(
      filename_legend_pairs,
      title=title,
      column='avg_latency',
      min_x=min_x,
      max_x=max_x,
      min_y=min_y,
      max_y=max_y,
      sigma=sigma)


def generate_p99_latency_graph(filename_legend_pairs,
                               title='p99 Latency',
                               min_x=0,
                               max_x=None,
                               min_y=0,
                               max_y=None,
                               sigma=0.1):
  """Generate p99 latency graph for the given CSV files.

    Args:
      filename_legend_pairs: CSV file path, or a pair of CSV file path and line
        legend title on the report.
      title: graph title.
      min_x: value to overwrite minimum X on the graph.
      max_x: value to overwrite maxiumum X on the graph.
      min_y: value to overwrite minimum Y on the graph.
      min_y: value to overwrite minimum Y on the graph.
      sigma: smoothing factor, the bigger value, the smother the graph.

    Returns:
      pyplot graph.
    """
  return generate_graph(
      filename_legend_pairs,
      title=title,
      column='p99',
      min_x=min_x,
      max_x=max_x,
      min_y=min_y,
      max_y=max_y,
      sigma=sigma)

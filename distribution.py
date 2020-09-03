import numpy as np

class Distribution(object):
  @staticmethod
  def factory(dist, rate):
    assert dist in Distribution.distributions()
    return globals()[dist.capitalize()](rate)

  @staticmethod
  def distributions():
    return [cls.__name__.lower() for cls in Distribution.__subclasses__()]

class Uniform(Distribution):
  def __init__(self, rate):
    self.rate = rate

  def next(self):
    return 1.0/self.rate

class Poisson(Distribution):
  def __init__(self, rate):
    self.rate = rate

  def next(self):
    return np.random.exponential(1/self.rate)

class Pareto(Distribution):
  def __init__(self, rate):
    self.rate = rate

  def next(self):
    # We chose the constant 2, so that the mean is (1/rate)
    return np.random.pareto(2) / self.rate

def test():
  p = 0
  u = 0
  e = 0

  rate = 100 # qps
  uniform = Uniform(rate)
  exp = Poisson(rate)
  pareto = Pareto(rate)

  for _ in range(100):
    u = uniform.next()
    e = exp.next()
    p = pareto.next()
    print("uniform: {:.2f} poisson: {:.2f} pareto: {:.2f}".format(u,e,p))

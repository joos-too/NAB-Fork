"""
Simple streaming baseline detectors for NAB.

These are implemented as NAB-native detectors (subclasses of AnomalyDetector),
so they can be run via `run.py` like any other detector.
"""
from collections import deque
import math
import os

from nab.detectors.base import AnomalyDetector



def _logisticScore(metric, center, scale):
  """
  Convert a non-negative anomaly metric (e.g. z-score) into a [0, 1] anomaly
  score. `center` controls where the score crosses 0.5.
  """
  if metric is None or not math.isfinite(metric):
    return 0.0

  if scale is None or scale <= 0:
    scale = 1.0

  x = (metric - center) / scale

  # Avoid exp() overflow.
  if x >= 60.0:
    return 1.0
  if x <= -60.0:
    return 0.0

  return 1.0 / (1.0 + math.exp(-x))



class ZScoreDetector(AnomalyDetector):
  """
  Sliding window Z-score detector.

  This detector outputs a smooth anomaly score using a logistic transform of
  the (absolute) z-score. The "threshold" parameter controls where the score
  crosses 0.5; NAB still optimizes a final threshold during the optimize step.
  """

  def __init__(self, *args, **kwargs):
    super(ZScoreDetector, self).__init__(*args, **kwargs)

    self.windowSize = int(os.environ.get("NAB_ZSCORE_WINDOW", "10"))
    self.threshold = float(os.environ.get("NAB_ZSCORE_THRESHOLD", "3.0"))
    self.scale = float(os.environ.get("NAB_ZSCORE_SCALE", "1.0"))
    self.minStd = float(os.environ.get("NAB_ZSCORE_MIN_STD", "1e-6"))

    self.window = deque(maxlen=self.windowSize)
    self._recordIndex = 0


  def handleRecord(self, inputData):
    score = 0.0
    value = inputData["value"]

    # Score using past-only statistics (do not include current point in window
    # stats, otherwise anomalies are diluted).
    if len(self.window) >= self.window.maxlen:
      mean = sum(self.window) / len(self.window)
      variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)
      std = math.sqrt(variance)
      if std < self.minStd:
        std = self.minStd

      z = abs((value - mean) / std)
      score = _logisticScore(z, center=self.threshold, scale=self.scale)

    self.window.append(value)

    if self._recordIndex < self.probationaryPeriod:
      score = 0.0

    self._recordIndex += 1
    return (score, )



class EwmaDetector(AnomalyDetector):
  """
  Exponentially Weighted Moving Average (EWMA) detector.

  The score is based on the standardized deviation from the EWMA, transformed
  via a logistic curve. `threshold` sets the 0.5 crossing point.
  """

  def __init__(self, *args, **kwargs):
    super(EwmaDetector, self).__init__(*args, **kwargs)

    self.alpha = float(os.environ.get("NAB_EWMA_ALPHA", "0.2"))
    self.threshold = float(os.environ.get("NAB_EWMA_THRESHOLD", "3.0"))
    self.scale = float(os.environ.get("NAB_EWMA_SCALE", "1.0"))
    self.minStd = float(os.environ.get("NAB_EWMA_MIN_STD", "1e-6"))

    self.ewma = None
    self.variance = 0.0
    self._recordIndex = 0


  def handleRecord(self, inputData):
    score = 0.0
    value = inputData["value"]

    if self.ewma is None:
      self.ewma = value
      self.variance = 0.0
    else:
      # Score against the previous EWMA (prediction), then update state.
      diff = value - self.ewma
      std = math.sqrt(self.variance)
      if std < self.minStd:
        std = self.minStd

      ratio = abs(diff) / std
      score = _logisticScore(ratio, center=self.threshold, scale=self.scale)

      self.ewma += self.alpha * diff
      self.variance = (1 - self.alpha) * (self.variance + self.alpha * diff * diff)

    if self._recordIndex < self.probationaryPeriod:
      score = 0.0

    self._recordIndex += 1
    return (score, )



class AdaptiveThresholdDetector(AnomalyDetector):
  """
  Adaptive threshold detector using a sliding mean and max deviation.

  Computes ratio = |x - mean| / max_dev. `sensitivity` controls where the
  logistic-transformed score crosses 0.5.
  """

  def __init__(self, *args, **kwargs):
    super(AdaptiveThresholdDetector, self).__init__(*args, **kwargs)

    self.windowSize = int(os.environ.get("NAB_ADAPTIVE_WINDOW", "20"))
    self.sensitivity = float(os.environ.get("NAB_ADAPTIVE_SENSITIVITY", "1.5"))
    self.scale = float(os.environ.get("NAB_ADAPTIVE_SCALE", "0.5"))
    self.minDev = float(os.environ.get("NAB_ADAPTIVE_MIN_DEV", "1e-6"))

    self.window = deque(maxlen=self.windowSize)
    self._recordIndex = 0


  def handleRecord(self, inputData):
    score = 0.0
    value = inputData["value"]

    # Score using past-only window statistics.
    if len(self.window) >= self.window.maxlen:
      mean = sum(self.window) / len(self.window)
      maxDev = max(abs(x - mean) for x in self.window)
      if maxDev < self.minDev:
        maxDev = self.minDev

      ratio = abs(value - mean) / maxDev
      score = _logisticScore(ratio, center=self.sensitivity, scale=self.scale)

    self.window.append(value)

    if self._recordIndex < self.probationaryPeriod:
      score = 0.0

    self._recordIndex += 1
    return (score, )

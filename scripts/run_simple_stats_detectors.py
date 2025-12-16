#!/usr/bin/env python3

import argparse
import os
import sys
try:
  import simplejson as json
except ImportError:
  import json

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

DETECTOR_NAMES = ["zScore", "ewma", "adaptiveThreshold"]


def _loadEnvFile(path):
  """
  Minimal .env reader: KEY=VALUE lines, optional leading `export `, ignores
  empty lines and comments. Missing files are ignored.
  """
  if not path or not os.path.exists(path):
    return

  with open(path) as f:
    for rawLine in f:
      line = rawLine.strip()
      if not line or line.startswith("#"):
        continue
      if line.startswith("export "):
        line = line[len("export "):].strip()
      if "=" not in line:
        continue

      key, value = line.split("=", 1)
      key = key.strip()
      value = value.strip()

      if (len(value) >= 2
          and value[0] == value[-1]
          and value[0] in ("'", '"')):
        value = value[1:-1]

      if key:
        os.environ[key] = value


def main(args):
  root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

  envFile = args.envFile
  if envFile and not os.path.isabs(envFile):
    envFile = os.path.join(root, envFile)
  _loadEnvFile(envFile)

  from nab.runner import Runner
  from nab.detectors.simple_stats.simple_stats_detectors import (
    ZScoreDetector,
    EwmaDetector,
    AdaptiveThresholdDetector)

  detectors = {
    "zScore": ZScoreDetector,
    "ewma": EwmaDetector,
    "adaptiveThreshold": AdaptiveThresholdDetector,
  }

  numCPUs = int(args.numCPUs) if args.numCPUs is not None else None

  dataDir = os.path.join(root, args.dataDir)
  windowsFile = os.path.join(root, args.windowsFile)
  resultsDir = os.path.join(root, args.resultsDir)
  profilesFile = os.path.join(root, args.profilesFile)
  thresholdsFile = os.path.join(root, args.thresholdsFile)

  runner = Runner(dataDir=dataDir,
                  labelPath=windowsFile,
                  resultsDir=resultsDir,
                  profilesPath=profilesFile,
                  thresholdPath=thresholdsFile,
                  numCPUs=numCPUs)

  runner.initialize()

  detectorNames = args.detectors
  unknown = sorted(set(detectorNames) - set(detectors.keys()))
  if unknown:
    raise SystemExit("Unknown detector(s): %s" % ", ".join(unknown))

  detectorConstructors = {name: detectors[name] for name in detectorNames}

  thresholds = None

  if args.detect:
    runner.detect(detectorConstructors)

  if args.optimize:
    thresholds = runner.optimize(detectorNames)

  if args.score:
    if thresholds is None:
      with open(thresholdsFile) as thresholdConfigFile:
        thresholds = json.load(thresholdConfigFile)
    runner.score(detectorNames, thresholds)

  if args.normalize:
    runner.normalize()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Run the simple statistics baseline detectors on NAB.")

  parser.add_argument("--envFile",
                    default=os.path.join("config", "simple_stats.env"),
                    help="Optional env file (KEY=VALUE) for detector tuning.")

  parser.add_argument("--detect",
                    default=False,
                    action="store_true",
                    help="Generate detector results.")

  parser.add_argument("--optimize",
                    default=False,
                    action="store_true",
                    help="Optimize thresholds per scoring profile.")

  parser.add_argument("--score",
                    default=False,
                    action="store_true",
                    help="Score results in the results directory.")

  parser.add_argument("--normalize",
                    default=False,
                    action="store_true",
                    help="Normalize the final scores.")

  parser.add_argument("-d", "--detectors",
                    nargs="*",
                    type=str,
                    default=DETECTOR_NAMES,
                    help="Detectors to run.")

  parser.add_argument("--dataDir",
                    default="data",
                    help="Directory containing NAB datasets.")

  parser.add_argument("--resultsDir",
                    default="results",
                    help="Directory to write/read detector results.")

  parser.add_argument("--windowsFile",
                    default=os.path.join("labels", "combined_windows.json"),
                    help="Ground truth labels JSON.")

  parser.add_argument("-p", "--profilesFile",
                    default=os.path.join("config", "profiles.json"),
                    help="Scoring profiles JSON.")

  parser.add_argument("-t", "--thresholdsFile",
                    default=os.path.join("config", "thresholds.json"),
                    help="Thresholds JSON.")

  parser.add_argument("-n", "--numCPUs",
                    default=None,
                    help="Number of CPUs to use (default: all).")

  args = parser.parse_args()

  if (not args.detect
      and not args.optimize
      and not args.score
      and not args.normalize):
    args.detect = True
    args.optimize = True
    args.score = True
    args.normalize = True

  if len(args.detectors) == 1:
    # Handle comma-separated list argument.
    args.detectors = args.detectors[0].split(",")

  main(args)

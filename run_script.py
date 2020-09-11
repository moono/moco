#!/usr/bin/python

import os

from subprocess import call

# set location
os.chdir('/work-dir')

# run python script
call(['python', '-u', 'main_moco.py',
      '--arch', 'resnet50',
      '--lr', '0.015',
      '--batch-size', '128',
      '--dist-url', "'tcp://localhost:10001'",
      '--multiprocessing-distributed',
      '--world-size', '1',
      '--rank', '0',
      '--mlp', '1',
      '--moco-k', '65536',
      '--moco-t', '0.2',
      '--aug-face',
      '--cos',
      '/dataset-dir'])

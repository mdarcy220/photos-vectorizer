#!/usr/bin/python3

import urllib.request
import os
import sys
import shutil

print('Downloading model (about 80 MB; this may take a while)... ', end='')
sys.stdout.flush()

with open(os.path.join(os.path.dirname(__file__), 'saved_model.dnn'), 'wb') as f:
	shutil.copyfileobj(urllib.request.urlopen('https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model'), f)

print('Done.')

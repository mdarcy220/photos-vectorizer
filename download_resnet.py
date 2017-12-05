#!/usr/bin/python3

# This file is part of Photos-vectorizer.
#
# Copyright (C) 2017  Mike D'Arcy
#
# Photos-vectorizer is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Photos-vectorizer is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import urllib.request
import os
import sys
import shutil

print('Downloading model (about 80 MB; this may take a while)... ', end='')
sys.stdout.flush()

with open(os.path.join(os.path.dirname(__file__), 'saved_model.dnn'), 'wb') as f:
	shutil.copyfileobj(urllib.request.urlopen('https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model'), f)

print('Done.')

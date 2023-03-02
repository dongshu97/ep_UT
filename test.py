import json
import os
import pathlib
from Tools import *

with open('.\config.json') as f:
  parsed_json = json.load(f)

print(parsed_json)

args = None

BASE_PATH, name = createPath(args)
if os.name != 'posix':
  prefix = '\\'
else:
  prefix = '/'

with open(BASE_PATH+prefix+"config.json", "w") as outfile:
  json.dump(parsed_json, outfile)
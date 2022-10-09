
import json
from os import mkdir

with open('dev/res/project-minimal.json') as prj_file:

  prj = json.loads(prj_file.read())

  required_keys = [
    'name', 'version'
  ]

  print('Setting up python project.')

  for key in required_keys: 

    prj[key] = input(f'{key}: ')

  with open('project.json', 'w') as target:

    print("Writing project.json file...")
    target.write(json.dumps(prj, indent=4))
  

  try:

    print('Creating source directory...')
    mkdir(prj['name'])

  except: 

    pass
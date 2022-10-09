#!/usr/bin/env python

from distutils.core import setup
import json

with open('./project.json') as project_file:

  prj = json.loads(project_file.read())

  setup(
    name=prj['name'],
    version=prj['version'],
    py_modules=prj['modules'],
  )
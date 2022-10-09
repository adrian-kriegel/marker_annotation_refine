#!/bin/sh

touch requirements.txt

python dev/create-project-json.py

sh ./dev/init.sh

#!/usr/bin/env bash

python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
ipython kernel install --user --name=orderdag-bounds
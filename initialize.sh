#!/bin/bash

# Creating the virtual env
python3 -m venv .venv

# Activating the virtual environment
source /home/tahmid/Projects/my-own/practice/ner-with-fastapi/.venv/bin/activate

pip install -r ./dependencies.txt

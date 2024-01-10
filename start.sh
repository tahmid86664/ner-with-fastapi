#!/bin/bash

# Activating the virtual environment
source /home/tahmid/Projects/my-own/practice/ner-with-fastapi/.venv/bin/activate 

uvicorn main:app --reload

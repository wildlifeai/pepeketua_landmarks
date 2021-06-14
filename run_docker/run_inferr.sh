#!/bin/bash
script="$0"
first="$1"
cd /code
if $first
then
  python main_inferr.py -d /info -s
else
  python main_inferr.py -d /info
fi
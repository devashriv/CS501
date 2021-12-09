#!/usr/bin/env bash
touch $1
for i in {1..30}
do
    python main.py >> $1
done

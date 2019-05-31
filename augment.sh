#!/bin/bash


ls data | while read dir ; do
    echo $dir
    python3 augmentation.py -n 4000 -c "./data/$dir/"
done

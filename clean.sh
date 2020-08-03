#!/bin/bash
torchserve --stop
rm -r model_store
rm -r logs
mkdir model_store
sh create_mar.sh
mv chordrecog.mar model_store

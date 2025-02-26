#!/bin/bash
dataset="popsign_v1_0"
category="game"
set_split="train"
words=("awake" "blow" "callonphone" "can" "dance" "cry" "cut" "drink" "drop" "find" "finish" "give" "go" "jump" "kiss" "listen" "look" "make" "ride" "say" "talk" "think" "touch" "wait" "wake")

for sign in "${words[@]}"; do
  echo "==========Fetching==========" \
  && echo "Dataset: $dataset, Category: $category, Set Split: $set_split, Sign: $sign" \
  && echo "----------Creating Appropriate Directory----------" \
  && mkdir -p "./$dataset/$category/$set_split/" \
  && echo "----------Downloading Tar File----------" \
  && wget -O "./$dataset/$category/$set_split/$sign.tar" "https://signdata.cc.gatech.edu/data/$dataset/$category/$set_split/$sign.tar" \
  && echo "----------Extracting Content----------" \
  && tar -xf "./$dataset/$category/$set_split/$sign.tar" -C "./$dataset/$category/$set_split/" \
  && echo "----------Deleting Original Tarfile" \
  && rm "./$dataset/$category/$set_split/$sign.tar" \
  && echo "==========Done=========="
done

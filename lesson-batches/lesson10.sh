#!/bin/bash
dataset="popsign_v1_0"
category="game"
set_split="train"
words=("backyard" "cloud" "clown" "cowboy" "fireman" "doll" "flag" "garbage" "gift" "hate" "hesheit" "minemy" "owie" "police" "pretend" "puzzle" "shhh" "sick" "story" "stuck" "weus" "grass" "rain" "snow" "sun" "tree" "flower" "outside" "home" "moon")

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

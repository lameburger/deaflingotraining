#!/bin/bash
dataset="popsign_v1_0"
category="game"
set_split="train"
words=("alligator" "animal" "bee" "bird" "bug" "cow" "donkey" "duck" "elephant" "fish" "frog" "giraffe" "hen" "horse" "kitty" "lion" "mouse" "owl" "pig" "puppy" "tiger" "wolf" "zebra" "cat" "dog" "goose")

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

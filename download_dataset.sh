#!/usr/bin/bash

dataset="popsign_v1_0"
category="game"
set_split="train"

# List of sign words for Lesson 1 (Family & People)
signs=(
    "grandma" "grandpa" "man" "mom" "uncle" "person"
)

echo "========== Fetching Dataset: $dataset, Category: $category, Set Split: $set_split =========="

# Create necessary directories
mkdir -p "./$dataset/$category/$set_split/"

# Loop through each sign and download it
for sign in "${signs[@]}"; do
    echo "---------- Processing: $sign ----------"
    sign_path="./$dataset/$category/$set_split/$sign.tar"
    
    echo "Downloading $sign..."
    wget -O "$sign_path" "https://signdata.cc.gatech.edu/data/$dataset/$category/$set_split/$sign.tar"
    
    echo "Extracting $sign..."
    tar -xf "$sign_path" -C "./$dataset/$category/$set_split/"
    
    echo "Deleting original tar file for $sign..."
    rm "$sign_path"
    
    echo "---------- Completed: $sign ----------"
done

echo "========== All Downloads Completed =========="

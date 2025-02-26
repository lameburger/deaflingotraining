local_folder="lesson7"
dataset_url="popsign_v1_0"
category="game"
set_split="train"
words=("drop" "find" "finish" "give" "go" "jump" "kiss" "listen" "look" "make" "ride" "say" "talk" "think" "touch" "wait" "wake")

for sign in "${words[@]}"; do
  echo "==========Fetching=========="
  echo "Local Folder: $local_folder, Category: $category, Set Split: $set_split, Sign: $sign"
  echo "----------Creating Appropriate Directory----------"
  mkdir -p "./$local_folder/$category/$set_split/"
  echo "----------Downloading Tar File----------"
  wget -O "./$local_folder/$category/$set_split/$sign.tar" "https://signdata.cc.gatech.edu/data/$dataset_url/$category/$set_split/$sign.tar"
  echo "----------Extracting Content----------"
  tar -xf "./$local_folder/$category/$set_split/$sign.tar" -C "./$local_folder/$category/$set_split/"
  echo "----------Deleting Original Tarfile----------"
  rm "./$local_folder/$category/$set_split/$sign.tar"
  echo "==========Done=========="
done

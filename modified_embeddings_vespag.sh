# parse the arguments 
while getopts ":e:i:o:" opt; do
  case $opt in
    e) embedding_folder="$OPTARG";;
    i) input_fasta="$OPTARG";;
    o) output_folder="$OPTARG";;
  esac
done

# check if the embedding folder is provided
if [ -z "$embedding_folder" ]; then
  echo "Please provide the embedding folder"
  exit 1
fi

# check if the input fasta file is provided
if [ -z "$input_fasta" ]; then
  echo "Please provide the input fasta file"
  exit 1
fi

# check if the output folder is provided
if [ -z "$output_folder" ]; then
  echo "Please provide the output folder"
  exit 1
fi

# make the output folder if it does not exist
mkdir -p $output_folder

# list all the files in the embedding folder
embedding_files=$(ls $embedding_folder)

# for each embedding file, run the vespag script
for embedding_file in $embedding_files
do
  # get the name of the embedding file
  file_name=$(basename $embedding_file .h5)
  # check if $output_folder/$file_name already exists and skip if yes 
  if [ -d $output_folder/$file_name ]; then
    echo "$output_folder/$file_name already exists. Skipping..."
    continue
  # else run the vespag script
  else
    python -m vespag predict -i $input_fasta -e $embedding_folder/$embedding_file -o $output_folder/$file_name
  fi
done


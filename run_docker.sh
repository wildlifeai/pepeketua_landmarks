docker run -it -v "$(pwd)"/new_models:/code/model -v "$(pwd)"/images:/info wildlife/frog_landmark:latest ./run_inferr.sh false

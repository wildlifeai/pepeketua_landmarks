@ECHO OFF
set image_dir_path=%1

robocopy %image_dir_path% "run_docker/dir" /E
echo Starting to build docker image with directory images
docker build run_docker -t archey:v1
echo Running prediction
docker run --name ArcheysFrogsLandMarkDetection archey:v1 sh ./run_inferr.sh

docker wait ArcheysFrogsLandMarkDetection
echo Copying prediction
docker cp ArcheysFrogsLandMarkDetection:/code/prediction.pkl prediction.pkl
echo Deleting docker container
docker rm ArcheysFrogsLandMarkDetection
echo Deleting docker image
docker image rm archey:v1
echo Deleting image dir
rd /S /Q "run_docker/dir"

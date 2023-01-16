@ECHO OFF
set image_dir_path=%1
set create_images=%2

robocopy %image_dir_path% "run_docker/dir" /E
echo Starting to build docker image with directory images
docker build run_docker -t archey:v2
echo Running prediction
if %create_images% equ true docker run --name ArcheysFrogsLandMarkDetection archey:v2 sh ./run_inferr.sh true
if %create_images% neq true docker run --name ArcheysFrogsLandMarkDetection archey:v2 sh ./run_inferr.sh false
docker wait ArcheysFrogsLandMarkDetection
echo Copying prediction
docker cp ArcheysFrogsLandMarkDetection:/code/prediction.pkl prediction.pkl
if %create_images% equ true echo Copying images
if %create_images% equ true docker cp ArcheysFrogsLandMarkDetection:/code/landmark_images/ landmark_images
echo Deleting docker container
docker rm ArcheysFrogsLandMarkDetection
echo Deleting docker image
docker image rm archey:v2
echo Deleting image dir
rd /S /Q "run_docker/dir"
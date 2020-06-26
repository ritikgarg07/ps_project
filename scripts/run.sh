IMAGE=ps_image
CONTAINER=ps_container
VERSION=0.01

docker run --mount type=bind,source=${PWD}/src,target=/workspaces/ --gpus all -it ${IMAGE}:${VERSION} 

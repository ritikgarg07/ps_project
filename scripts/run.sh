IMAGE=ps_image
CONTAINER=ps_container
VERSION=0.01



docker run --mount type=bind,source=${PWD}/src,target=/src --gpus all -it ${IMAGE}:${VERSION} 

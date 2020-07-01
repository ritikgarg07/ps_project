IMAGE=ps_image

docker run --mount type=bind,source=${PWD},target=/workspaces/ --gpus all -it ${IMAGE}

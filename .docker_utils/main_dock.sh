#!/bin/bash 

XAUTH=$HOME/.Xauthority

# Determine the directory where the script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Use a relative path for the data directory
DATA_DIR="$SCRIPT_DIR/../"
echo "Data directory mounting: $DATA_DIR"

# Check if the first argument is 'cuda' or 'jetson'
GPU_OPTION=""
EXTRA_MOUNTS=""
if [ "$1" == "cuda" ]; then
    GPU_OPTION="--gpus all"
elif [ "$1" == "jetson" ]; then
    GPU_OPTION="--gpus all --runtime=nvidia"
    EXTRA_MOUNTS="--volume /run/jtop.sock:/run/jtop.sock"
fi

# Check if a second argument is provided for the container name, else use default
CONTAINER_NAME="${2:-embodiedai_dock}"

docker run --privileged -it \
  --env DISPLAY=$DISPLAY \
  --net=host \
  --volume /dev:/dev \
  --volume $XAUTH:/root/.Xauthority \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume="$DATA_DIR:/embodiedai" \
  $EXTRA_MOUNTS \
  --name="$CONTAINER_NAME" \
  --detach \
  $GPU_OPTION \
  --rm \
  --shm-size=1gb \
  embodiedai:latest \
  /bin/bash

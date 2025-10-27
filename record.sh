#!/bin/bash

BAGS=$HOME/opencv-cuda-plus/bags/$(date +%m%d_%H%M)

if [ ! -d "$BAGS" ]; then
    mkdir -m 775 $BAGS
else
    BAGS=$BAGS"$(date +%S)"
    if
        [ ! -d "$BAGS" ]; then
        mkdir -m 775 $BAGS
    fi
fi

BAGS=$BAGS"/"
echo "BAGS: "$BAGS
# Create a directory with current date and time as its name



# Record bag files with auto-splitting
rosbag record -o $BAGS --split --size=2048 -b 2048 \
/camera_pano_masked/image_raw/compressed
# /camera1/color/image_raw/compressed \
# /camera2/color/image_raw/compressed \
# /camera3/color/image_raw/compressed \
# /camera4/color/image_raw/compressed \
# /js/velodyne_points \
# /inference/image_udt/compressed

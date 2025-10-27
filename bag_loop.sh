while true; do
    for bagfile in "$1"/*.bag; do
        echo "--- Now playing: $bagfile ---"
        rosbag play "$bagfile"
        sleep 5
    done
    echo "All bag files played. Restarting in 3 seconds..."
    sleep 3
done
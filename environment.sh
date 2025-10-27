source /opt/ros/noetic/setup.bash
source $HOME/opencv-cuda-plus/catkin_ws/devel/setup.bash


if [ $# -gt 0 ]; then
	export ROS_MASTER_IP=$1
    echo "ROS_MASTER_IP set to $ROS_MASTER_IP"
    source set_ros_master.sh $ROS_MASTER_IP
else
    source set_ros_master.sh 192.168.10.133
    # source set_ros_master.sh 172.20.10.2
fi

if [ $# -gt 0 ]; then
	export ROS_IP=$2
    echo "ROS_IP set to $ROS_IP"
    source set_ros_ip.sh $ROS_IP
else
    source set_ros_ip.sh 192.168.10.136
    # source set_ros_ip.sh 172.20.10.2
fi


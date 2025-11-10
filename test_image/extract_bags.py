import os
import cv2
import argparse
from cv_bridge import CvBridge
import rosbag
import numpy as np

class ArgBagExtractor:
    def __init__(self, bag_file, output_dir, image_topic, video_output_file, accelerate_rate, compressed): 
        self.bag_file = bag_file
        self.bag = rosbag.Bag(self.bag_file, "r")
        self.output_dir = output_dir
        self.image_topic = image_topic
        self.video_output_file = video_output_file
        self.accelerate_rate = accelerate_rate
        self.compressed = compressed
        self.fps = self.calculate_fps()
        self.frame_skip_interval = int(self.fps / self.accelerate_rate) if self.accelerate_rate > 0 else 1

    def calculate_fps(self):
        """Calculate FPS based on the timestamps in the ROS bag file."""
        timestamps = []
        for _, msg, t in self.bag.read_messages(topics=[self.image_topic]):
            timestamps.append(t.to_sec())
        
        if len(timestamps) < 2:
            return 10  # Default FPS if timestamps are insufficient
        
        # Calculate the average time difference between frames
        avg_time_diff = np.mean(np.diff(timestamps))
        calculated_fps = 1.0 / avg_time_diff if avg_time_diff > 0 else 10
        print(f"Calculated FPS: {calculated_fps:.2f}")
        return calculated_fps

    def extract(self):
        bridge = CvBridge()
        count = 0
        frame_skip = 0
        video_writer = None

        for topic, msg, t in self.bag.read_messages(topics=[self.image_topic]):
            # Skip frames based on calculated frame_skip_interval
            if frame_skip < self.frame_skip_interval:
                frame_skip += 1
                continue
            frame_skip = 0
            count += 1

            # Decode the image based on raw or compressed format
            if self.compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

            # Save the image
            image_path = os.path.join(self.output_dir, f"{count}.png")
            cv2.imwrite(image_path, cv_img)

            # Initialize the video writer if it's not already done
            if video_writer is None:
                height, width = cv_img.shape[:2]
                video_writer = cv2.VideoWriter(
                    self.video_output_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,  # Use calculated FPS for the video
                    (width, height)
                )

            # Write the frame to the video
            video_writer.write(cv_img)

        # Release video writer and close bag
        if video_writer:
            video_writer.release()
        self.bag.close()
        print(f"Done, {count} images saved and video output to {self.video_output_file}")

def main(args):
    bags = os.listdir(args.bag_dir)
    # Set default topics and output names if none are provided
    image_topic_list = args.topic if args.topic else ["/camera1/color/image_raw", "/camera2/color/image_raw", "/camera3/color/image_raw"]
    output_names = args.output_names if args.output_names else ["_left", "_mid", "_right"]

    for bag in bags:
        bag_file = os.path.join(args.bag_dir, bag)
        print(bag_file)

        output_dir_list = [os.path.join(args.output_image_dir, f"{bag[:-4]}{name}/") for name in output_names]
        video_output_files = [os.path.join(args.output_video_dir, f"{bag[:-4]}{name}.mp4") for name in output_names]

        os.makedirs(args.output_video_dir, exist_ok=True)

        for output_dir, image_topic, video_output_file in zip(output_dir_list, image_topic_list, video_output_files): 
            print(f"Saving images to {output_dir} and video to {video_output_file}")
            os.makedirs(output_dir, exist_ok=True)
            extractor = ArgBagExtractor(bag_file, output_dir, image_topic, video_output_file, args.accelerate_rate, args.compressed)
            extractor.extract()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and create video from ROS bag files.")
    parser.add_argument("--bag_dir", type=str, default="bags", help="Directory containing bag files.")
    parser.add_argument("--output_image_dir", type=str, default="images", help="Directory to save extracted images.")
    parser.add_argument("--output_video_dir", type=str, default="videos", help="Directory to save output videos.")
    parser.add_argument("--accelerate_rate", type=float, default=1.0, help="Rate to accelerate playback (fps/rate).")
    parser.add_argument("--topic", type=str, nargs='*', default=None, help="Image topic(s) to extract. Defaults to '/camera1/color/image_raw', '/camera2/color/image_raw', '/camera3/color/image_raw'.")
    parser.add_argument("--output_names", type=str, nargs='*', default=None, help="Names for each output directory corresponding to topics. Defaults to '_left', '_mid', '_right'.")
    parser.add_argument("--compressed", action="store_true", help="Use compressed image format.")
    
    args = parser.parse_args()
    # print(args.topic)
    # print(args.output_names)
    main(args)

# Example usage:

# python3 extract_bags.py --accelerate_rate 5.0 --topic /inference/image_udt/compressed --output_names _test --compressed

# python3 extract_bags.py --accelerate_rate 5.0 --topic /camera1/color/image_raw/compressed /camera2/color/image_raw/compressed /camera3/color/image_raw/compressed /camera_pano_masked/image_raw/compressed --output_names _left _mid _right _pano --compressed
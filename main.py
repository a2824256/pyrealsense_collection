import png
import pyrealsense2 as rs
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import os

def make_directories():
    if not os.path.exists("JPEGImages/"):
        os.makedirs("JPEGImages/")
    if not os.path.exists("depth/"):
        os.makedirs("depth/")

if __name__ == "__main__":
    make_directories()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    intr = color_frame.profile.as_video_stream_profile().intrinsics

    align_to = rs.stream.color
    align = rs.align(align_to)
    number = 0
    while True:

        filecad = "JPEGImages/%s.jpg" % number

        filedepth = "depth/%s.png" % number
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        cv2.imshow('COLOR IMAGE', c)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(filecad, c)
            writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                bitdepth=16, greyscale=True)
            with open(filedepth, 'wb') as f:
                zgray2list = d.tolist()
                writer.write(f, zgray2list)
            number += 1

    cv2.destroyAllWindows()

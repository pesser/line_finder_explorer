#!/usr/bin/env python

"""Interactive Exploration of parameters for Line Detection."""

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import time

TOL = 1e-6
N_FLAGS = 4 # number of flags that control image generation, see make_img


def apply_mask(img, vertices):
    """Apply mask to image if vertices of mask are defined."""
    if vertices is None:
        return img
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    imshape = img.shape
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines, color = (255, 0, 0), thickness = 2):
    """Draw lines onto img."""
    if thickness > 0:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def angle(a, b):
    """Return angle between two vectors."""
    return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) * 180 / np.pi


def filter_lines(lines, angle_tolerance = 20):
    """Filter lines for lane detection.

    Only consider lines of a certain angle and split between lines
    indicating left and right lanes.
    """
    left_lines = []
    right_lines = []

    for line in lines:
        left_lane_lines = []
        right_lane_lines = []
        for px, py, qx, qy in line:
            p = np.array([px, py])
            q = np.array([qx, qy])
            s = q - p

            # normalize to have an upwards pointing line segment
            if s[1] < 0:
                s *= -1

            left_lane_prior = np.array([1.0, 1.0])
            right_lane_prior = np.array([-1.0, 1.0])
            if angle(s, left_lane_prior) < angle_tolerance:
                left_lane_lines.append((p[0], p[1], q[0], q[1]))
            if angle(s, right_lane_prior) < angle_tolerance:
                right_lane_lines.append((p[0], p[1], q[0], q[1]))
        if left_lane_lines:
            left_lines.append(left_lane_lines)
        if right_lane_lines:
            right_lines.append(right_lane_lines)

    return left_lines, right_lines


def lines_to_lanes(lines, y_bottom, y_middle):
    """Infer lanes from the given lines.
    
    Extrapolate line segments such that they go from y_bottom to y_middle on
    the y component and then take the median of the x components of the
    points defining the line.
    """
    lane_bottoms = []
    lane_middles = []

    for line in lines:
        for px, py, qx, qy in line:
            lambda_middle = (y_middle - py) / (qy - py)
            lambda_bottom = (y_bottom - py) / (qy - py)

            x_middle = np.round(px + lambda_middle * (qx - px)).astype(int)
            x_bottom = np.round(px + lambda_bottom * (qx - px)).astype(int)

            lane_bottoms.append(x_bottom)
            lane_middles.append(x_middle)

    # take the medians for final decision and return a result that is
    # compatible with the structure returned by HoughLinesP, i.e. a sequence
    # of lines each being a sequence of x1,y1,x2,y2 where (x1,y1) and
    # (x2, y2) determine the line.
    lane = []
    if lane_bottoms and lane_middles:
        x_bottom = np.round(np.median(lane_bottoms)).astype(int)
        x_middle = np.round(np.median(lane_middles)).astype(int)
        lane.append([(x_bottom, y_bottom, x_middle, y_middle)])

    return lane


def detect_lle(
        img,
        gauss_kernel_size,
        canny_low_threshold, canny_high_threshold,
        hough_rho, hough_theta, hough_threshold,
        hough_min_line_length, hough_max_line_gap,
        angle_tolerance,
        vertical_lane_offset = 60,
        mask_vertices = None,
        **kwargs):
    """Find lanes, lines and edges of img."""
    # make sure gaussian kernel size is odd
    if gauss_kernel_size % 2 == 0:
        gauss_kernel_size += 1
    # find edges
    edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.GaussianBlur(edges, (gauss_kernel_size, gauss_kernel_size), 0)
    edges = cv2.Canny(edges, canny_low_threshold, canny_high_threshold)

    # mask
    edges = apply_mask(edges, mask_vertices)

    # find lines
    lines = cv2.HoughLinesP(
            edges,
            hough_rho, hough_theta * np.pi/180,
            hough_threshold,
            np.array([]),
            hough_min_line_length, hough_max_line_gap)
    # replace with empty iterable if no lines were found
    lines = lines if lines is not None else []

    # filter the lines found
    left_lines, right_lines = filter_lines(lines, angle_tolerance)
    
    # infer lanes from line segments
    left_lane = lines_to_lanes(left_lines, img.shape[0], img.shape[0] // 2 + vertical_lane_offset)
    right_lane = lines_to_lanes(right_lines, img.shape[0], img.shape[0] // 2 + vertical_lane_offset)

    return left_lane, right_lane, left_lines, right_lines, lines, edges


def make_img(img, mode, **kwargs):
    """Make line image for the given img.
    
    kwargs are passed to detect_lle,
    mode is a bit flag to determine what is drawn:
    0-th bit: If set, use original image as background otherwise use edges
    1-th bit: If set draw lines detected in the image
    2-th bit: If set draw lanes detected in the image
    3-th bit: If set draw filtered lines
    """
    left_lane, right_lane, left_lines, right_lines, lines, edges = detect_lle(img = img, **kwargs)

    background_bit = 0
    lines_bit = 1
    lanes_bit = 2
    filtered_bit = 3

    # use either the original image or the edges of the image as canvas
    if mode & 1<<background_bit:
        base = np.copy(img)
    else:
        base = np.dstack(3*[edges])

    # draw lines and lanes on blank canvas
    bg = np.zeros_like(img)

    if mode & 1<<lines_bit:
        draw_lines(bg, lines, color = (255,0,0), thickness = 2)

    if mode & 1<<lanes_bit:
        draw_lines(bg, left_lane, color = (0,0,255), thickness = 6)
        draw_lines(bg, right_lane, color = (0,0,255), thickness = 6)

    if mode & 1<<filtered_bit:
        draw_lines(bg, left_lines, color = (255,0,255), thickness = 6)
        draw_lines(bg, right_lines, color = (0,255,255), thickness = 6)

    # combine base with lines
    return cv2.addWeighted(base, 0.8, bg, 1.0, 0.0)


def make_mask(image,
        mask_horizontal_aperture = 30,
        mask_vertical_adjustment = 50):
    """Generate vertices to be used as a mask.
    
    Bottom vertices are at the bottom corners of the image, the other two
    vertices are at the center of the image adjusted by the two parameters.
    """
    height, width = image.shape[:2]
    vertices = np.array([[
        (0, height - 1),
        ((width - 1) // 2 - mask_horizontal_aperture,
            (height - 1) // 2 + mask_vertical_adjustment),
        ((width - 1) // 2 + mask_horizontal_aperture,
            (height - 1) // 2 + mask_vertical_adjustment),
        (width - 1, height - 1)]],
        dtype=np.int32)
    return vertices


if __name__ == "__main__":
    import sys
    import itertools

    if len(sys.argv) < 2:
        print("Useage: {} input_image.jpg [input_image2.jpg ...]".format(sys.argv[0]))
        print("Cycle through images with 'n'. Exit with 'q'.")
        exit(1)

    fnames = sys.argv[1:]
    video_fnames = [fname for fname in fnames if fname.endswith(".mp4")]
    img_fnames = [fname for fname in fnames if fname.endswith(".jpg")]

    # read images
    images = []
    for fname in img_fnames:
        images.append(cv2.imread(fname))
    videos = []
    for fname in video_fnames:
        videos.append(VideoFileClip(fname))

    print("Read {} images and {} videos.".format(len(images), len(videos)))

    # Line finder
    input_cycle = itertools.cycle(images + videos)
    current_input = next(input_cycle)
    current_frame = 0
    #vertices = make_mask(image)
    state = {
            "img": None,
            "result": None,
            "gauss_kernel_size": 1,
            "canny_low_threshold": 50,
            "canny_high_threshold": 100,
            "hough_rho": 1,
            "hough_theta": 1,
            "hough_threshold": 30,
            "hough_min_line_length": 10,
            "hough_max_line_gap": 5,
            "angle_tolerance": 20,
            "mask_vertices": None,
            "mask_horizontal_aperture": 80,
            "mask_vertical_adjustment": 80,
            "mode": int("0101", 2),
            "stop": False}

    # parameters exposed to ui together with maximum values
    ui_max_params = {
            "gauss_kernel_size": 50,
            "canny_low_threshold": 500,
            "canny_high_threshold": 500,
            "hough_threshold": 500,
            "hough_min_line_length": 500,
            "hough_max_line_gap": 500,
            "mask_horizontal_aperture": 500,
            "mask_vertical_adjustment": 500,
            "mode": (1 << N_FLAGS) - 1}

    # Create window to show result and controls
    cv2.namedWindow("Line Detection", 0)

    def update(param, k, state):
        state[k] = param
        state["result"] = None

    for k in sorted(ui_max_params):
        cv2.createTrackbar(
                k, "Line Detection",
                state[k],
                ui_max_params[k],
                lambda param, k = k, state = state: update(param, k, state))


    # Main loop
    while True:
        if type(current_input) != np.ndarray:
            if current_frame == 0:
                fps = current_input.fps
                current_input = itertools.cycle(current_input.iter_frames(dtype = "uint8"))
            if not state["stop"]:
                frame = next(current_input).copy()
                frame[:,:,:] = frame[:,:,[2,1,0]]
                time.sleep(1.0/fps)
                update(frame, "img", state)
                update(make_mask(
                    frame,
                    mask_horizontal_aperture = state["mask_horizontal_aperture"],
                    mask_vertical_adjustment = state["mask_vertical_adjustment"]),
                    "mask_vertices", state)
                current_frame += 1
        else:
            if current_frame == 0:
                update(current_input, "img", state)
                update(make_mask(current_input), "mask_vertices", state)
                update(make_mask(
                    current_input,
                    mask_horizontal_aperture = state["mask_horizontal_aperture"],
                    mask_vertical_adjustment = state["mask_vertical_adjustment"]),
                    "mask_vertices", state)
                current_frame += 1


        if state["result"] is None:
            state["result"] = make_img(**state)

        cv2.imshow("Line Detection", state["result"])

        key = cv2.waitKey(1)
        if  key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next input
            current_input = next(input_cycle)
            current_frame = 0
        elif key == ord('s'):
            # toggle stop for videos
            state["stop"] = not state["stop"]

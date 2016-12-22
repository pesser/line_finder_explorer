#!/usr/bin/env python

"""Interactive Exploration of parameters for Line Detection."""

import numpy as np
import cv2

TOL = 1e-6
N_FLAGS = 3 # number of flags that control image generation, see make_img


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


def lines_to_lanes(lines, y_bottom, y_middle):
    """Infer lanes from the given lines.
    
    Filter out horizontal line segments, extrapolate line segment such that
    it goes from y_bottom to y_middle on the y component, seperate line
    segments for left lane from those for right lane and finally take the
    median of the x components of the points defining the line.
    """
    left_lane_bottoms = []
    left_lane_middles = []
    right_lane_bottoms = []
    right_lane_middles = []

    for line in lines:
        px, py, qx, qy = line[0]
        length = np.sqrt((qx - px)**2 + (qy - py)**2)
        # only accept lines that are neither vertical nor horizontal
        if((np.abs(qx - px) / length > TOL) and
                (np.abs(qy - py) / length > TOL)):
            lambda_middle = (y_middle - py) / (qy - py)
            lambda_bottom = (y_bottom - py) / (qy - py)

            x_middle = np.round(px + lambda_middle * (qx - px)).astype(int)
            x_bottom = np.round(px + lambda_bottom * (qx - px)).astype(int)

            # decide which lane this line corresponds to by looking at the
            # quadrant in which the line is
            if (qx > px and qy > py) or (qx < px and qy < py):
                # first or third quadrant is left lane
                left_lane_bottoms.append(x_bottom)
                left_lane_middles.append(x_middle)
            else:
                # second and fourth quadrant is right lane
                right_lane_bottoms.append(x_bottom)
                right_lane_middles.append(x_middle)

    # take the medians for final decision and return a result that is
    # compatible with the structure returned by HoughLinesP, i.e. a sequence
    # of lines each being a sequence of x1,y1,x2,y2 where (x1,y1) and
    # (x2, y2) determine the line.
    lanes = []
    if left_lane_bottoms:
        x_bottom_left = np.round(np.median(left_lane_bottoms)).astype(int)
        x_middle_left = np.round(np.median(left_lane_middles)).astype(int)
        lanes.append([(x_bottom_left, y_bottom, x_middle_left, y_middle)])
    if right_lane_bottoms:
        x_bottom_right = np.round(np.median(right_lane_bottoms)).astype(int)
        x_middle_right = np.round(np.median(right_lane_middles)).astype(int)
        lanes.append([(x_bottom_right, y_bottom, x_middle_right, y_middle)])

    return lanes


def detect_lle(
        img,
        gauss_kernel_size,
        canny_low_threshold, canny_high_threshold,
        hough_rho, hough_theta, hough_threshold,
        hough_min_line_length, hough_max_line_gap,
        vertical_lane_offset = 50,
        mask_vertices = None, **kwargs):
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
    
    # infer lanes from line segments
    lanes = lines_to_lanes(lines, img.shape[0], img.shape[0] // 2 + vertical_lane_offset)

    return lanes, lines, edges


def make_img(img, mode, **kwargs):
    """Make line image for the given img.
    
    kwargs are passed to detect_lle,
    mode is a bit flag to determine what is drawn:
    0-th bit: If set, use original image as background otherwise use edges
    1-th bit: If set draw lines detected in the image
    2-th bit: If set draw lanes detected in the image
    """
    lanes, lines, edges = detect_lle(img = img, **kwargs)

    background_bit = 0
    lines_bit = 1
    lanes_bit = 2

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
        draw_lines(bg, lanes, color = (0,0,255), thickness = 6)

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

    # read images
    images = []
    for fname in fnames:
        images.append(cv2.imread(fname))
    print("Read {} images.".format(len(images)))

    # Line finder
    image_cycle = itertools.cycle(images)
    image = next(image_cycle)
    vertices = make_mask(image)
    state = {
            "img": image,
            "result": None,
            "gauss_kernel_size": 3,
            "canny_low_threshold": 50,
            "canny_high_threshold": 150,
            "hough_rho": 1,
            "hough_theta": 1,
            "hough_threshold": 40,
            "hough_min_line_length": 5,
            "hough_max_line_gap": 5,
            "mask_vertices": vertices,
            "mode": int("101", 2)}
    vertices = make_mask(image)

    # parameters exposed to ui together with maximum values
    ui_max_params = {
            "gauss_kernel_size": 50,
            "canny_low_threshold": 500,
            "canny_high_threshold": 500,
            "hough_threshold": 500,
            "hough_min_line_length": 500,
            "hough_max_line_gap": 500,
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
        if state["result"] is None:
            state["result"] = make_img(**state)

        cv2.imshow("Line Detection", state["result"])

        key = cv2.waitKey(1)
        if  key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next image
            image = next(image_cycle)
            update(image, "img", state)
            update(make_mask(image), "mask_vertices", state)

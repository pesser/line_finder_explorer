#!/usr/bin/env python
# Interactive Exploration of parameters for Line Detection.

import numpy as np
import cv2

class LineFinder(object):
    """ Find Lines in image using
    - Mask
    - Blur
    - Canny edge detector
    - Hough Transform
    """

    def __init__(self, in_img, vertices = None):
        """ Initialize with input image and mask vertices. """
        # store image and grayscale version of it
        self.image = in_img
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # vertices of mask - ignored if None
        self.vertices = vertices

        # default parameters
        self.default_params = {
                "gauss_kernel_size": 0,
                "canny_low_threshold": 1,
                "canny_high_threshold": 1,
                "hough_rho": 1,
                "hough_theta": 1,
                "hough_threshold": 1,
                "hough_min_line_length": 1,
                "hough_max_line_gap": 1,
                "apply_mask": 0,
                "line_thickness": 0}
        # parameters exposed to ui together with maximum values
        self.max_params = {
                "gauss_kernel_size": 50,
                "canny_low_threshold": 500,
                "canny_high_threshold": 500,
                "hough_threshold": 500,
                "hough_min_line_length": 500,
                "hough_max_line_gap": 500,
                "apply_mask": 1,
                "line_thickness": 50}
        self.params = self.default_params.keys()
        self.update_params(**self.default_params)
               

    def _apply_mask(self, img):
        """ apply mask to image if vertices of mask are defined. """
        if self.vertices is None:
            return img
        mask = np.zeros_like(img)
        ignore_mask_color = 255
        imshape = img.shape
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)


    def _draw_lines(self, lines):
        """ draw lines from probabilistic Hough Detector. """
        line_image = np.zeros_like(self.image)
        line_color = (255, 0, 0)
        if self.line_thickness > 0:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, self.line_thickness)
        return line_image


    def update_params(self, **kwargs):
        """ Update parameters and recreate line image with these parameters.
        """
        for k, v in kwargs.items():
            assert(k in self.params)
            setattr(self, k, v)
        self.make_img()


    def make_img(self):
        """ Make line image. """
        # GaussianBlur required an odd kernel size
        kernel_size = 2*self.gauss_kernel_size + 1
        blurred = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), 0)

        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        if self.apply_mask:
            edges = self._apply_mask(edges)

        lines = cv2.HoughLinesP(
                edges,
                self.hough_rho, self.hough_theta * np.pi/180,
                self.hough_threshold,
                np.array([]),
                self.hough_min_line_length, self.hough_max_line_gap)
        # replace with empty iterable if no lines were found
        lines = lines if lines is not None else []

        line_image = self._draw_lines(lines)

        color_edges = np.dstack((edges, edges, edges)) 
        self.lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 


    def get_img(self):
        return self.lines_edges


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Useage: {} input_image.jpg".format(sys.argv[0]))
        exit(1)

    # Read in image
    image = cv2.imread(sys.argv[1])

    # Define mask
    height, width = image.shape[:2]
    vertices = np.array([[
        (0, height - 1),
        ((width - 1) // 2 - 30, (height - 1) // 2 - 25),
        ((width - 1) // 2 + 3,  (height - 1) // 2 - 25),
        (width - 1, height - 1)]],
        dtype=np.int32)
    # Initialize Line Finder
    updater = LineFinder(image, vertices)

    # Create window to show result and controls
    cv2.namedWindow("Line Detection", 0)
    for k in sorted(updater.max_params):
        cv2.createTrackbar(k, "Line Detection", updater.default_params[k], updater.max_params[k],
                lambda param, k = k: updater.update_params(**{k: param}))

    # Main loop
    while True:
        lines_edges = updater.get_img()
        cv2.imshow("Line Detection", lines_edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

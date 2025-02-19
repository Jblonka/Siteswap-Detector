﻿# Siteswap-Detector

## About
The Siteswap Detector is a Python-based tool designed to analyze juggling patterns from video footage. Using computer vision and object tracking techniques, the tool identifies and classifies juggling patterns, commonly represented in siteswap notation.


## Usage
There are 3 versions of the Siteswap-Detector available:

1. hsv_range:
    A simple approach to ball detection. Given an hsv range where the ball color falls into, this algorithm locates the juggling balls in the frame. To know what the approximate hsv range of your juggling balls is use the `hsv_color_picker.py` file.

    This algorithm is quick, but not very accurate.

2. background_subtraction:
    This approach adds to the previous one by adding a background subtraction feature. Here the background of the video gets removed, helping in reducing detection errors.

    This algorithm is slower, but more accurate than the previous one.

3. histogram_backprojection:
    In this approach the user selects a ball in the first frame of the video. The color of that ball is used to find the others in the concurrent frames. Background subtraction is also implemented.

    This algorithm is quicker than the 2nd one. The accuracy depends on the quality of the ball selection.


Change into the directory of the version you want to use. The run the main script and provide the path to the video file as an argument. Also provide the number of balls that are juggled. Optionally you can add the path to where the result video should be saved.

`python main.py --video <path_to_video> --max_balls <number_of_balls> --output_path <path_to_output_video>`

There are a few videos available you can use in this google drive folder: https://drive.google.com/drive/folders/1QWsv04h2yCowf5f7AqeA0Q7H4FcvL2ZI?usp=sharing

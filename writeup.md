## Project 4 - Advanced Lane Finding report
---

The code for this project can be found in `advanced_lane_detection.ipynb`. The code snippets provided in this document refer to the same file.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard.png "Chessboard Transformation"
[image2]: ./output_images/road.png "Road Transformed"
[image3]: ./output_images/thresholded_image.png "Binary Example"
[image4]: ./output_images/warped_image.png "Warp Example"
[image5]: ./output_images/masked_image.png "Mask Example"
[image6]: ./output_images/mask_warp_image.png "Warp after Mask Example"
[image7]: ./output_images/pipeline.png "Warp after Mask Example"
[image8]: ./output_images/histogram.png "Histogram Example"
[image9]: ./output_images/polyfit.png "Histogram Example"
[image10]: ./output_images/lane_image.png "Lane Image Example"
[image11]: ./output_images/lane_radius_deviation_image.png "Lane with radius and deviation value superimposed Image Example"

---
### Camera Calibration

Chessboard images are used to calibrate the camera. Chessboard images are captured from the same camera as the project video. `def find_corners(imgs, x, y)` function detects the corners in chessboard. The number of corners to be detected are provided through *no_x* and *no_y*. Chessboard images can be found in the folder *camera_cal*. Chessboard corners can be found with the help of *cv2.findChessboardCorners*, which takes in gray scale images and returns a flag if the required number of corners were found along with coordinate values of all the corners. According to the flag received, the images are classified for further analysis.

For each calibration image, 3D points and 2D point coordinates are computed. 3D points are real world, 3D space coordinates with z-axis values being 0. 3D points and 2D points are appended to a list for each calibration image.

The camera calibration and distortion correction values are obtained through `cv2.calibrateCamera`. This function takes in 2D points, 3D points and gray scale image.

With the help of `pickle`, calibration values are saved for calibrating the images that are obtained from the vehicle.

### Pipeline (single images)

#### 1. Distortion-correction.
Calibration and distortion correction values obtained through `cv2.calibrateCamera`, are used to correct the distortion in an image. `cv2.undistort` function is used to obtain an undistorted image.

Below is an image of original chessboard image, and road captured from camera and undistorted image formed with the help of openCV.
![alt text][image1]
![alt text][image2]

#### 2. Thresholding to generate a binary image
This project mainly aims at determing the lane lines on road. The lane lines are marked with yellow color or white color and this is contrasting with respect to road. To filter out on what is needed, color and gradient thresholding are performed.

`def convert_to_binary(img, sobel_thresh=(25,255), satu_thresh=(125, 255), light_thresh=(125,255))` is the function call for binary image generation.

The input image is converted to HLS color space, and H(color) channel is discarded, since we are mainly concentrating on contrast difference. Light and Saturation channels are considered for this project. The threshold levels are determined by trial and error method. Sobel operator is used for contrast detection which is a derivative. The derivative will be high if the contrast difference between two points are more.

A bitwise 'and' for the three masks provided a binary image as below for one of the test image.

![alt text][image3]

#### 3. Perspective transform
The code for my perspective transform can be found in the function call `def img_warp(img, top_view=True)`.  The `img_warp()` function takes as inputs an image (`img`), and a flag for top view which determines if Bird's view is required or if inverse transformation from top view to normal view is required. Source (`src`) and destination (`dst`) points are hardcode in the following manner:

```python
src = np.float32(
    [[200, 720],
     [585, 450],
     [700, 450],
     [1160, 720]])
dst = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]])
```

The mapping of source and destination is as shown below.

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 200, 720        |
| 585, 450      | 200, 0      |
| 700, 450     | 1080, 0      |
| 1160, 720      | 1080, 720        |

The below image shows a top view of with the defined `src` and `dst` points.

![alt text][image4]

#### 4. Region of Interest
Function `def img_msk(img)` is used to mask the region outside the lane boundary. The masked image is then passed to perspective view. This helps in blocking other additional objects in image. Below shows an image of masking and then warping a test image.

![alt text][image5]
![alt text][image6]

The entire pipeline of undistorting, converting to binary, masking and warping is as shown below.

![alt text][image7]

#### 5. Drawing Lane lines with the help of Polyfit

Histogram `def histogram_peaks(img)` of the pipelined image is plotted to better estimate the lane. Larger the vertical line represents there are lanes. A plot of histogram is as shown below.

![alt text][image8]

The two peaks represents the lanes, the corresponding x-axis represents the pixels. The lane detection and a polyfit is defined in the function `def sliding_window(img)`. The histogram is divided into half, with the left part indication left lane and the right part indicating right lane. A margin is defined and the index values within the margin are stored if the pixel values are non zero. These index values are then fed to a second order polynomial of the form (ax**2 + bx +c) with the help of `np.polyfit` function. The below image shows the polyfit of a lane line (Red in color).

![alt text][image9]

#### 6. Marking the Entire Lane
`def draw_lane(org_img, msk_bin_img, left_fit, right_fit)` is used to draw Lane marking. X and Y coordinates are stacked together to be used in `cv2.fillPoly` function. Once the Lane is drawn on the binary pipelined image, the entire image is inverse warped and superimposed on the original image as shown below.

![alt text][image10]

#### 7. Radius of Curvature and deviation from center
`def radius_of_curvature(ploty, left_lane_inds, right_lane_inds, img_shape)`

From [chatbotslife](https://chatbotslife.com/advanced-lane-line-project-7635ddca1960),

Radius of curvature =​​ (1 + (dy/dx)^2)^1.5 / abs(d2y /dx2) and x is given by a second order polynomial, x = ay^2 + by + c. Taking derivatives of this gives the radius as (1 + (2a y_eval+b)^2)^1.5 / abs(2a).

The image is represented as pixels. To calculate the radius in meters, we have to convert each pixel in terms of meters. This is calculated by the following,
```python
y_met_per_pix = 30 / 720 # meters per pixel in y dimension
x_met_per_pix = 3.7 / 700 # meters per pixel in x dimension
```

By applying the formula, radius of curvature is determined.

To calculate the deviation of car from center, we assume that the camera is mounted exactly in the center of car. The center of left and right lane line are determined and the difference between car image center and lane center produces the deviation result.

#### 8. Result plotted back down onto the road such that the lane area is identified clearly

`def processing_pipeline(img)` provides an overview on how the radius of curvature text and deviation from center text are superimposed on the lane detected image. Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Link to the output video.

Here's a [link to my video result](./project_video_output.mp4). This also estimates the radius of curvature and the deviation from center.

---

### Discussion

#### References
[chatbotslife](https://chatbotslife.com/advanced-lane-line-project-7635ddca1960): Has described on how to implement advanced lane line detection and also has provided formulae required in this project.

#### Further Improvements

The pipeline that I have implemented does not store the previous values. This helps in faster tracking. This pipeline is not tested on challenge videos and needs more work. I would like to work on this in future to improve and optimize the code to work on all the videos.

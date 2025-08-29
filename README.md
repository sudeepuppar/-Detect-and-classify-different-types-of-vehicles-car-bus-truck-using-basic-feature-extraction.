# -Detect-and-classify-different-types-of-vehicles-car-bus-truck-using-basic-feature-extraction.
This Python project detects and classifies vehicles (bike, car, truck) from a video feed using OpenCV. It applies a bird’s eye view transformation to better track and analyze vehicle motion within a defined Region of Interest (ROI).


Features:

 Manual ROI selection via 4-point mouse click
 Perspective transformation to top-down view
 Background subtraction using KNN (with shadow detection disabled)
 Vehicle detection using contour analysis
 Vehicle classification into: bike,car,truck
 

Requirements:
 Python 3.x
 OpenCV (cv2)
 NumPy

 Install dependencies using:
 pip install opencv-python numpy


Select ROI:
 Click 4 points on the first frame in the order:
 Top-left
 Top-right
 Bottom-right
 Bottom-left

These define the area for the perspective transformation.
View Detection:
 A top-down "bird’s eye view" window with detections
 Original video with the selected ROI marked
 Foreground mask used for motion detection
 Press Esc to exit.

Vehicle Classification Logic
Vehicles are classified by bounding box area and aspect ratio:

 if area > 50000:
    return "truck"
 elif area > 7000:
    return "Car"
 else:
    return "bike"

Future Improvements:
 Add vehicle tracking with IDs (e.g., using SORT or Deep SORT)
 Improve classification using a trained ML model
 Count vehicles crossing a line
 Export analytics (vehicle count per type, speed, etc..)




 

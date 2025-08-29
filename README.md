# -Detect-and-classify-different-types-of-vehicles-car-bus-truck-using-basic-feature-extraction.
This Python project detects and classifies vehicles (bike, car, truck) from a video feed using OpenCV. It applies a bird’s eye view transformation to better track and analyze vehicle motion within a defined Region of Interest (ROI).


🚗 Vehicle Detection with Bird’s Eye View Perspective

## 📸 Features

- Manual ROI selection via 4-point mouse click  
- Perspective transformation to top-down view  
- Background subtraction using KNN (with shadow detection disabled)  
- Vehicle detection using contour analysis  
- Vehicle classification into:
  - **Bike**
  - **Car**
  - **Truck**

---

## 🧰 Requirements

- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  

### 📦 Install dependencies:

```bash
pip install opencv-python numpy
```

---

## 🖱️ Select ROI

Click **4 points** on the first video frame in the order:

1. Top-left  
2. Top-right  
3. Bottom-right  
4. Bottom-left  

These define the area for the perspective transformation.

---

## 🎥 View Detection

- A top-down **bird’s eye view** window with real-time vehicle detection  
- Original video with the selected ROI highlighted  
- Foreground mask showing motion-based areas  
- Press `Esc` to exit the program

---

## 🧠 Vehicle Classification Logic

Vehicles are classified using bounding box **area** and **aspect ratio**:

```python
if area > 50000:
    return "truck"
elif area > 7000:
    return "Car"
else:
    return "bike"
```

Adjust these thresholds in the `classify_vehicle()` function for better results depending on your video input.

---

## 🔮 Future Improvements

- Add vehicle tracking with unique IDs (e.g., using **SORT** or **Deep SORT**)  
- Improve classification using a trained **machine learning** or **deep learning** model  
- Count vehicles crossing a virtual line or zone  
- Export analytics (e.g., vehicle count per type, estimated speed, timestamps, etc.)

---

## 📄 License

This project is open-source and licensed under the **MIT License**.

 

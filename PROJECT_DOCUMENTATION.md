# CrowdMind AI - Project Documentation

## Project Overview
CrowdMind AI is a real-time crowd analysis and risk assessment system. It uses computer vision and deep learning to estimate crowd density, count people in video feeds, and predict the risk of stampedes or congestion. The project features a Streamlit-based dashboard for live visualization, including heatmaps and zone-wise statistics.

---

## Hardware Utilization
The system is designed to automatically detect and use **GPU (CUDA)** if available for high-performance neural network inference.
- **GPU (CUDA):** Used for CSRNet (Density Estimation) and CrowdRiskClassifier (Risk Assessment) when a compatible NVIDIA GPU is detected.
- **CPU:** Used as a fallback if no GPU is available or if CUDA is not installed.
- **Memory Management:** The system includes manual garbage collection (`gc.collect()`) and GPU cache clearing (`torch.cuda.empty_cache()`) to ensure stability during continuous video processing.

---

## File Structure & Responsibilities

### 1. [app.py](file:///c:/Users/Ayush/OneDrive/Desktop/Projects/CrowdMind%20AI/app.py)
**Role:** The main UI layer built with Streamlit.
- **Functionality:** 
    - Handles video uploading and stream control.
    - Manages the UI layout (Live Feed, Heatmap, Live Stats).
    - Contains the rendering loop that interacts with `processor.py`.

### 2. [model.py](file:///c:/Users/Ayush/OneDrive/Desktop/Projects/CrowdMind%20AI/model.py)
**Role:** Central repository for neural network architectures.
- **`CrowdRiskClassifier` (Class):** CNN for risk classification.
- **`CSRNet` (Class):** Dilated CNN for density estimation.
- **`make_layers` (Function):** Standardized layer builder for both models.

### 3. [processor.py](file:///c:/Users/Ayush/OneDrive/Desktop/Projects/CrowdMind%20AI/processor.py)
**Role:** Core processing engine.
- **`DensityEstimationCounter` (Class):** Manages CSRNet inference and counting.
- **`CrowdMindProcessor` (Class):** Coordinates counting, risk assessment, and memory management.
- **Functionality:** Pre-processes frames, runs models, and returns analysis metrics.

---

## Key Functions Detail

| Function | File | Description |
| :--- | :--- | :--- |
| `count_people` | `processor.py` | Runs CSRNet to generate density maps and people counts. |
| `assess_risk` | `processor.py` | Uses CrowdRiskClassifier to predict risk levels. |
| `process_frame` | `processor.py` | The main pipeline that executes the full analysis loop. |
| `preprocess_frame` | `processor.py` | Prepares images for neural network input. |
| `get_crowd_processor` | `app.py` | Efficiently loads and caches the processor for the UI. |

---

## How it Works (Data Flow)
1. **Input:** Video is captured from a file or webcam via OpenCV.
2. **Analysis:** 
   - Frame is sent to **CSRNet** → Output: Density Heatmap + Count.
   - Frame is sent to **Risk Classifier** → Output: Risk Level (Low/Med/High).
3. **UI Update:** 
   - The Heatmap is overlayed on the original video.
   - Metrics (Total count, Zone counts) are updated in the sidebar.
   - If risk is High, a warning alert is triggered.
4. **Optimization:** Torch memory is cleared after every few frames to prevent buildup.

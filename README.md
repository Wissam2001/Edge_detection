# Edge Detection from Scratch: Laplacian of Gaussian & Canny Algorithms 🔍
This project was developed as a Master's Lab project in Image Processing and Computer Vision, focused on understanding and implementing edge detection algorithms from scratch. Instead of relying on OpenCV's built-in functions, we manually implemented two fundamental edge detection techniques:
- **Laplacian of Gaussian (LoG):** A second-order derivative approach combining Gaussian smoothing with Laplacian edge detection
- **Canny Edge Detector:** A multi-stage algorithm widely considered the optimal edge detecton
The project features an interactive GUI built with Tkinter that allows users to upload images, adjust parameters, and visualize the edge detection results in real-time.

# Project Objectives 🎯
- **Understand edge detection theory:** Master the mathematical foundations of gradient-based and second-derivative edge detection
- **Implement LoG from scratch:** Create a Laplacian of Gaussian filter with zero-crossing detection
- **Implement Canny algorithm from scratch:** Build all five stages:
- **Gaussian smoothing for noise reduction**
- **Gradient calculation using Sobel operators**
- **Non-maximum suppression for edge thinning**
- **Double thresholding (hysteresis) for edge linking**
- **Build an interactive GUI:** Allow users to experiment with parameters (sigma, kernel size, thresholds)
- **Compare edge detection methods:** Visualize the differences between LoG and Canny outputs

# Libraries Used 🛠️
- **tkinter:**	GUI creation (file dialogs, buttons, input fields)
- **PIL (Pillow):**	Image loading, resizing, and format conversion
- **numpy:**	Matrix operations, convolutions, array manipulations
- **scipy.ndimage:**	Additional image processing utilities
- **matplotlib:**	(Imported but not heavily used - for potential visualizations)

# Lessons Learned 💡
**1. Edge Detection Theory**
- LoG detects edges at zero-crossings of the second derivative → good for finding edges regardless of orientation
- Canny is more robust to noise and produces cleaner, thinner edges
- Gaussian smoothing is critical before derivative operations to avoid amplifying noise

**2. Implementation Challenges**
- Convolution with padded arrays - Must handle boundaries correctly to avoid shrinking the image
- Zero-crossing detection: Requires checking multiple directions to avoid false positives
- Non-maximum suppression: Gradient direction must be quantized to 4 discrete angles
- Hysteresis thresholding: Requires recursive/iterative edge linking logic

**3. Parameter Sensitivity**
- Sigma (σ)	Larger values = more smoothing = fewer, thicker edges
- Kernel size	Should be ~6σ+1 to properly represent Gaussian
- Thresholds	Higher = fewer edges (only strong gradients)
- Low/High ratio	Typically 1:3 for good edge linking

**Example Parameters:**
Algorithm	Recommended Values
- LoG	Sigma: 1.0-2.0, Kernel: 5-9, Threshold: 10-50
- Canny	Sigma: 1.0, Kernel: 5, High threshold: 100, Low threshold: 30-50

 # License 📝
  This project is for educational purposes as part of a Master's lab assignment.

# Contact ✉️
- **Email:** wissambadia4@gmail.com
- **LinkedIn:** [Badia Ouissam Lakas](linkedin.com/in/badia-ouissam-lakas-a66a28214) 




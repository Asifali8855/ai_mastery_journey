Paragraph 1: Common Setup and Dependency Errors
This project highlighted frequent challenges tied to environment configuration and dependency management. 
The initial errors stemmed from Python version incompatibility (e.g., TensorFlow not supporting Python 3.13),
requiring downgrades to Python 3.11/3.12. Missing dependencies like matplotlib, opencv-python, or tensorflow caused
ModuleNotFoundError, resolved via pip install.
Paragraph 2: Runtime and File Handling Issues
Runtime errors included incorrect file paths (e.g., my_digit.png not found) and
missing model files (mnist_digit_model.h5), solved by verifying paths or re-training the model. 
OpenCV’s cv2.resize() failed when images were unreadable due to typos, permissions, or corrupted files. 
Preprocessing steps (e.g., grayscale conversion, resizing) had to mimic MNIST’s 28x28 format, and TensorFlow warnings 
about oneDNN optimizations or metric compilation were benign but required code adjustments. Finally, ensuring error handling
(e.g., if img is None) and validating dependencies upfront streamlined debugging. These issues underscored the importance of
meticulous path management, dependency checks, and testing incremental steps.

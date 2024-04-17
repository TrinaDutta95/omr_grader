This is a project I did as a TA for a Graduate course. The professor wanted me to write a python program which will automatically grade a scantron page filled by students.
# Scantron Grading System

This project provides an automated solution to grade scantron sheets using image processing techniques with Python and OpenCV. The system processes scanned images of filled-in scantron forms to determine which bubbles have been marked and evaluates the answers based on a provided answer key.

## Features

- **Image Processing**: Utilizes OpenCV to handle image manipulations and contour detection.
- **Adaptive Thresholding**: Helps in accurate bubble detection under varying lighting conditions in scan images.
- **Contour Analysis**: Identifies filled bubbles and compares them to an answer key.
- **Result Visualization**: Marks correct answers in green and incorrect ones in red on the scanned image.

## Prerequisites

Before you can run this project, you'll need to install the following:

- Python 3.6 or higher
- OpenCV-Python
- NumPy
- Imutils

You can install these packages using pip:

```bash
pip install requirements.txt
```
## Project Structure
- main.py: Main script that includes all the image processing and grading logic.
- images/: Directory containing sample scantron images to test the grading system (which is not included for confidential reason)
- .pdf :Pdf file containing scantrons (which is not included for confidential reason)
- README.md: This file, providing an overview and instructions for the project.

## Usage
To use this scantron grading system, you will need to provide:
-A path to the scanned pdf file of the filled scantron in the main.py.
-An answer key in the form of a dictionary where the key is the question number (starting from 0) and the value is the index of the correct answer (starting from 0) in the main.py
```bash
python main.py
```
- Empty the "images/" folder before every run otherwise, it will encounter an error
  
## To run this project on Custom Scantron
- Make sure to update the crop section of the code
  ```bash
  # Define the box to crop (left, upper, right, lower)
    box = (30, 300, 120, 590)  # Adjust these values based on your observation
  ```
- Make sure to update the bubble contour section of the code
  ```bash
  # Print the geometric properties
        if w >= 8 and h >= 8 and circularity >= 0.7 and ar <= 1.1:
  ```
  These values will vary depending on the scanned file and the size of the bubbles. So you may need to adjust them for your custom page.

## Output
The script will process the image, evaluate the answers, and save a new image showing the results with correct answers. Additionally, it will print the score and a detailed report on the terminal. An example output:
```bash
Image with detected bubbles saved as quiz3/page_10_bubbles.png
(10, [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```
  

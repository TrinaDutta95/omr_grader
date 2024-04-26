from imutils import contours
import numpy as np
from PIL import Image
import cv2


# Process each scanned page for grading
def process_scantron(image_path):
    # Read the image using OpenCV
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Error loading image")
        return

    # answers
    answer_key = {0:1,1:3,2:0,3:1,4:0,5:3,6:1,7:1,8:2,9:1,10:3,11:1,12:0,13:2,14:3} # replace with answer keys for different quizzes
    # Convert OpenCV image to a PIL image for cropping
    img_pil = Image.fromarray(img_cv)
    # Define the box to crop (left, upper, right, lower)
    box = (30, 300, 120, 655)  # Adjust these values based on your observation
    cropped_image = img_pil.crop(box)
    # Convert the cropped PIL image back to an OpenCV image
    img_cv_cropped = np.array(cropped_image)

    # Save the cropped image (optional, for checking the crop)
    cv2.imwrite(image_path.replace('.png', '_cropped.png'), img_cv_cropped)

    # Preprocess the cropped image
    image = img_cv_cropped

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 5)

    cv2.imwrite(image_path.replace('.png', '_thresh.png'), thresh)

    # Find contours
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    questionCnts = []
    # Analyze each contour
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)

        # Calculate circularity if perimeter is not zero to avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        # print("w:",w,"h:",h,"circularity:",circularity,"ar:",ar)
        # Print the geometric properties
        if w >= 8 and h >= 8 and circularity >= 0.6 and ar <= 1.1:
            questionCnts.append(c)

    output_img = cv2.cvtColor(img_cv_cropped, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawing
    for cnt in questionCnts:
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)  # Draw contours in green
    save_path = image_path.replace('.png', '_bubbles.png')  # Modify path for saving
    cv2.imwrite(save_path, output_img)
    print(f"Image with detected bubbles saved as {save_path}")

    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts,
                                          method="top-to-bottom")[0]
    correct = 0
    score_board = []
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        # loop over the sorted contours to find the filled bubble
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        k = answer_key[q]
        # check to see if the bubbled answer is correct
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
            score_board.append(1)
        else:
            score_board.append(0)

        # draw the outline of the correct answer on the test
        #cv2.drawContours(img_cv_cropped, [cnts[k]], -1, color, 3)
        #cv2.imwrite(image_path.replace('.png', '_correct.png'), img_cv_cropped)
    # grab the test taker
    score = correct
    return score, score_board
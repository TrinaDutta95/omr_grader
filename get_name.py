import numpy as np
from PIL import Image
import string
import cv2


# Filtering bubble contours

def validate_contours(cnts):
    filtered_contours = []
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        fill_ratio = area / (w * h)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Calculate circularity if perimeter is not zero to avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        # print("w:", w, "h:", h, "circularity:", circularity, "ar:", aspect_ratio)
        # Check if contour is roughly circular and has reasonable size and fill ratio
        if w >= 8 and h >= 8:
            filtered_contours.append(c)

    return filtered_contours


# Parsing name from contours
def parse_names(image_path):
    # Read the image using OpenCV
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Error loading image")
        return None

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cv = clahe.apply(img_cv)

    # Convert OpenCV image to a PIL image for cropping
    img_pil = Image.fromarray(img_cv)
    box = (122, 52, 433, 291)  # Adjust these values based on your observation
    cropped_image = img_pil.crop(box)

    # Rotate the image by 270 degrees clockwise
    rotated_image = cropped_image.rotate(270, expand=True)
    if rotated_image.mode != 'RGB':
        rotated_image = rotated_image.convert('RGB')

    rotated_image_cv = np.array(rotated_image)
    rotated_image_cv = cv2.cvtColor(rotated_image_cv, cv2.COLOR_RGB2GRAY)

    # Save the cropped and rotated image
    save_path = image_path.replace('.png', '_cropped_rotated.png')
    cv2.imwrite(save_path, rotated_image_cv)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(rotated_image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    cv2.imwrite(image_path.replace('.png', '_thresh.png'), thresh)

    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_cnts = validate_contours(cnts)
    # print("Number of contours found:", len(valid_cnts))
    # Create a blank image of the same dimensions as the original
    output_img = cv2.cvtColor(rotated_image_cv, cv2.COLOR_GRAY2BGR)
    # Draw each contour
    for contour in valid_cnts:
        cv2.drawContours(output_img, [contour], -1, 255, 2)  # Draw white contours

    # Optionally, save the image
    # cv2.imwrite('bubble_contours.png', output_img)
    # Sort contours; first by y then by x (this needs to be refined based on actual image layout)
    cnts = sorted(valid_cnts, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))
    sorted_contours_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for i, ctr in enumerate(valid_cnts):
        cv2.drawContours(sorted_contours_img, [ctr], -1, (0, 255, 0), 1)
    # cv2.imwrite(image_path.replace('.png', '_sorted_contours.png'), sorted_contours_img)

    alphabet = string.ascii_uppercase
    num_columns = 20
    num_rows = 26  # We know there are 26 rows for letters A-Z

    results = ['_'] * num_columns

    # Calculate width of each column based on the sorted contours
    first_contour_bounds = cv2.boundingRect(cnts[0])
    last_contour_bounds = cv2.boundingRect(cnts[-1])
    scantron_width = last_contour_bounds[0] + last_contour_bounds[2] - first_contour_bounds[0]
    column_width = scantron_width / num_columns
    # print(scantron_width/column_width)
    # Process each column
    start_x = first_contour_bounds[0]
    for col in range(num_columns):
        column_contours = [cnts[i] for i in range(len(cnts)) if
                           cv2.boundingRect(cnts[i])[0] >= start_x + col * column_width and
                           cv2.boundingRect(cnts[i])[0] < start_x + (col + 1) * column_width]
        max_pixels = 0
        selected_letter = None
        selected_contour = None  # To hold the contour of the most filled bubble

        # Sort column contours by y to ensure correct row ordering
        column_contours = sorted(column_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

        for i, c in enumerate(column_contours):
            if i >= 26:  # Ensure we don't exceed the alphabet list
                break
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            # print(alphabet[i], total)
            if total > max_pixels and total < 100:
                max_pixels = total
                selected_letter = alphabet[i]  # Match contour index directly to alphabet
                selected_contour = c
        # print(selected_letter, max_pixels)
        # Update the result for this column with the letter having maximum filled bubble
        if selected_letter is not None and max_pixels > 65:
            results[col] = selected_letter
            # Draw the most filled bubble contour in red
            color_img = cv2.drawContours(rotated_image_cv, [selected_contour], -1, (0, 0, 255), 2)

    # Save the image with marked bubbles
    # cv2.imwrite('marked_scantron.png', color_img)

    print("Detected names:", results)
    return results

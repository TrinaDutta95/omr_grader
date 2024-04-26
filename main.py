import os
import fitz  # PyMuPDF
from get_name import validate_contours, parse_names
from grader import process_scantron


def convert_pdf_to_images(pdf_path, output_folder):
    # Open the provided PDF
    doc = fitz.open(pdf_path)

    # Iterate through each page
    for i, page in enumerate(doc):
        # Render page to an image (pix)
        pix = page.get_pixmap()
        image_path = f"{output_folder}/page_{i + 1}.png"
        # if first or second page is empty then delete that and adjust this part of code
        print(i)
        if (i+1)%2 == 0:
            pix.save(image_path)
            print(f"Saved {image_path}")

    # Close the document
    doc.close()


if __name__ == '__main__':
    # processing the pdf file to individual images into a folder
    convert_pdf_to_images('quiz3.pdf', 'quiz3')  # change file and folder accordingly
    # take each image from processed folder and run it for grading
    folder_dir = "quiz3/"  # change directory accordingly
    for images in os.listdir(folder_dir):
        print(images)
        # if student has not filled bubble or incorrectly filled bubble e.g.-crossing or tick, it will not register
        # as a filled bubble
        name = parse_names(folder_dir+images)
        name_string = ' '.join([str(item) for item in name])
        print(name_string)
        print(process_scantron(folder_dir+images))

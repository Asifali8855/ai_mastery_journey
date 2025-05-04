import cv2
import pytesseract
import os
import time

# ✅ SET THE CORRECT PATH TO YOUR TESSERACT EXECUTABLE
pytesseract.pytesseract.tesseract_cmd = r'C:\ai_practice_days\day6\tesseract.exe'

# ✅ SET YOUR IMAGE FOLDER PATH
WATCH_FOLDER = r"C:\ai_practice_days\day6\day10\input_images"

def preprocess_image(image_path):
    """
    Convert image to grayscale and apply thresholding
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_text(image):
    """
    OCR to extract text using pytesseract
    """
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=config)

def process_new_images():
    """
    Detect new images in the folder and process them
    """
    processed_files = set()

    while True:
        for file in os.listdir(WATCH_FOLDER):
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and file not in processed_files:
                file_path = os.path.join(WATCH_FOLDER, file)
                print(f"\n🖼️ Processing: {file}")
                preprocessed = preprocess_image(file_path)
                text = extract_text(preprocessed)
                print(f"📄 Extracted Text:\n{text}")
                processed_files.add(file)
        time.sleep(5)  # scan folder every 5 seconds

if __name__ == "__main__":
    print("📂 Watching folder for new handwritten images...")
    process_new_images()

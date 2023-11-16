import cv2
import numpy as np
import base64
import re
import easyocr
import openai
from fuzzywuzzy import fuzz

openai.api_key = '' 
reader = easyocr.Reader(['en', 'ja'], gpu=False)

section_heights = {'NSBM': 1010, 'UMO': 720}

def read_input(image_data): 
    nparr_answer = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr_answer, cv2.IMREAD_COLOR)
    return img

def divide_image(image, split_height):
    height, width = image.shape[:2]

    # Ensure the split height is within the valid range
    split_height = max(0, min(split_height, height))

    upper_part = image[:split_height, :]
    lower_part = image[split_height:, :]
    return upper_part, lower_part

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    horizontal_projection = np.sum(thresholded, axis=1)
    max_projection_value = np.max(horizontal_projection)
    threshold_fraction = 0.999 
    threshold = max_projection_value * threshold_fraction

    segment_start = None
    segments = []

    for i, projection_value in enumerate(horizontal_projection):
        if projection_value < threshold:
            if segment_start is None:
                segment_start = i
        else:
            if segment_start is not None:
                # Check if the current segment is significant enough (e.g., greater than a certain height)
                if i - segment_start >= 3:  
                    rgb_segment = cv2.cvtColor(cv2.cvtColor(thresholded[segment_start:i, :], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                    segments.append(rgb_segment)
                segment_start = None
    return segments

def text_in_segment(image_path):  
    results = reader.readtext(image_path)
    # Accessing the confidence score for the first result (assuming only one text region)
    confidence_score = round(results[0][2], 2)
    extracted_text = ' '.join([result[1] for result in results]) 
    return extracted_text, confidence_score

def remove_digits_and_symbols(text):
    # Use regex to remove digits and symbols
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def check_correctness_with_openai(text, marks, correct_mark):
    prompt = f"Fact check, Include true or false in the answer: \"{text}\"\n\nEvaluation:"
    response = openai.Completion.create(
        engine="text-davinci-003",  
        prompt=prompt,
        temperature=0.7,  
        max_tokens=50,  
        n=1
    )
    correctness = response.choices[0].text.strip()
    if 'True' in correctness:
        marks += correct_mark
        return correctness, True, marks
    else:
        marks = marks
        return correctness, False, marks
    
def encode_images(image):
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')
    return encoded_image

def add_symbol(image, is_checkmark):
    height, width, _ = image.shape 
    image_with_symbol = image.copy()

    # Calculate the coordinates for the symbol based on position
    x_position = int(0.8 * width)
    y_position = int(height / 2)

    if is_checkmark:
        # Draw a red checkmark
        checkmark_coordinates = np.array([[x_position - 40, y_position + 20],
                                          [x_position - 20, y_position + 45],
                                          [x_position + 45, y_position - 35]], dtype=np.int32)
        cv2.polylines(image_with_symbol, [checkmark_coordinates], isClosed=False, color=(0, 0, 255), thickness=2)
    else:
        # Draw a red cross
        cross_size = 25
        cv2.line(image_with_symbol, (x_position - cross_size, y_position - cross_size),
                 (x_position + cross_size, y_position + cross_size), color=(0, 0, 255), thickness=2)
        cv2.line(image_with_symbol, (x_position - cross_size, y_position + cross_size),
                 (x_position + cross_size, y_position - cross_size), color=(0, 0, 255), thickness=2)

    return image_with_symbol

def combine_color_images_vertically(image_list, white_space_rows=10):
    # Ensure all images have the same width
    min_width = min(image.shape[1] for image in image_list)
    # Resize images to have the same width
    resized_images = [cv2.resize(image, (min_width, int(image.shape[0] * (min_width / image.shape[1])))) for image in image_list]

    # Create a white row
    white_row = np.ones((white_space_rows, min_width, 3), dtype=np.uint8) * 255  # White color

    # Initialize the combined image with the first image
    combined_image = resized_images[0]

    # Add white rows and subsequent images
    for image in resized_images[1:]:
        combined_image = np.vstack([combined_image, white_row, image])

    return combined_image

def create_text_image(text, font=cv2.FONT_HERSHEY_COMPLEX, font_size=1, font_thickness=2, image_size=(400, 50), color=(255, 0, 0), background_color=(255, 255, 255)):
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
    image[:, :] = background_color

    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    text_position = ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2)

    cv2.putText(image, text, text_position, font, font_size, color, font_thickness)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def are_texts_similar(text1, text2, correct_mark, marks, threshold=80):
    similarity_ratio = fuzz.token_sort_ratio(text1, text2)
    if similarity_ratio >= threshold:
        marks += correct_mark
        return True, marks, similarity_ratio
    else:
        marks = marks
        return False, marks, similarity_ratio
    
def binary_to_rgb(input_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the binary image to an RGB image
    rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

    return rgb_image
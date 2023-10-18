import cv2
import numpy as np
import base64
import re
import easyocr
import openai
from PIL import Image

openai.api_key = '' 
reader = easyocr.Reader(['en'], gpu=False)

def read_input(image_data): 
    nparr_answer = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr_answer, cv2.IMREAD_COLOR)
    return img

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
                if i - segment_start >= 3:  # Adjust the height threshold as needed
                    segments.append(thresholded[segment_start:i, :])
                segment_start = None
    return segments

def text_in_segment(image_path):  
    results = reader.readtext(image_path)
    # Accessing the confidence score for the first result (assuming only one text region)
    confidence_score = results[0][2]
    extracted_text = ' '.join([result[1] for result in results]) 
    return extracted_text, confidence_score

def remove_digits_and_symbols(text):
    # Use regex to remove digits and symbols
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def check_correctness_with_openai(text, marks):
    prompt = f"Fact check, Include true or false in the answer: \"{text}\"\n\nEvaluation:"
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use an appropriate OpenAI engine
        prompt=prompt,
        temperature=0.7,  # Adjust as needed
        max_tokens=50,  # Adjust as needed
        n=1
    )
    correctness = response.choices[0].text.strip()
    if 'True' in correctness:
        marks += 5
        return correctness, True, marks
    else:
        marks = marks
        return correctness, False, marks
    
def encode_images(image):
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')
    return encoded_image

# def combine_images_vertically(image_arrays):
#     # Convert NumPy arrays to PIL images
#     images = [Image.fromarray(array) for array in image_arrays]
#     # Assuming all images are the same size
#     width, height = images[0].size
#     # Create a new image with the same width and combined height
#     combined_image = Image.new('L', (width, height * len(images)))
#     # Paste the images vertically
#     for i, image in enumerate(images):
#         combined_image.paste(image, (0, height * i))
#     combined_image_array = np.array(combined_image)
#     return combined_image_array

def add_text_to_binary_image(binary_image, is_correct):
    # Choose the color based on correctness
    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green for correct, Red for incorrect
    # Choose the text based on correctness
    display_text = 'Correct' if is_correct else 'Wrong'
    # Choose the font and size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 2
    font_thickness = 2
    # Calculate text position based on image size
    height, width = binary_image.shape[:2]
    text_position = (int(0.75 * width), int(height / 2))  # (x, y) coordinates
    # Create a copy of the binary image to avoid modifying the original
    image_with_text = binary_image.copy()
    # Add text to the binary image
    image_with_text = cv2.putText(image_with_text, display_text, text_position, font, font_size, color, font_thickness)
    return image_with_text

def combine_color_images_vertically(image_list):
    # Ensure all images have the same width
    min_width = min(image.shape[1] for image in image_list)
    # Resize images to have the same width
    resized_images = [cv2.resize(image, (min_width, int(image.shape[0] * (min_width / image.shape[1])))) for image in image_list]
    # Stack images vertically
    combined_image = np.vstack(resized_images)
    return combined_image

def create_text_image(text, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1, font_thickness=2, image_size=(500, 100), color=(255, 255, 255), background_color=(0, 0, 0)):
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
    image[:, :] = background_color

    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    text_position = ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2)

    cv2.putText(image, text, text_position, font, font_size, color, font_thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
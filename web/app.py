from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
import openai
import easyocr
import time

openai.api_key = ''
reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)

def encode_images(image):
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(encoded_image).decode('utf-8')
    return encoded_image

def read_text_from_image(image_path):  
    results = reader.readtext(image_path)
    extracted_text = ' '.join([result[1] for result in results]) 
    return extracted_text

def rearrange_text_with_chatgpt(text):
    prompt = f"The given text is not in order. Rearrange the text in correct grammatical order and it should definitely start with the number. It should only include the details provided by the input text. No additional details are needed:\n\n{text}\n\nReordered text: "
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can experiment with different engines
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def compare_similarity(text1, text2, marks):
    prompt = f"Compare the similarity between the following two texts. Check whether the idea of {text1} is identical to the idea of {text2}. Give the final result as a percentage:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use other engines as well
        prompt=prompt,
        max_tokens=50
    )
    
    similarity_result = response.choices[0].text.strip()
    if "100%" in similarity_result:
        marks += 5
        return similarity_result, True, marks
    else:
        marks = marks
        return similarity_result, False, marks

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    horizontal_projection = np.sum(thresholded, axis=1)

    # Calculate the maximum pixel value in the horizontal projection
    max_projection_value = np.max(horizontal_projection)

    # Define a threshold value as a fraction of the maximum projection value
    threshold_fraction = 0.999 
    threshold = max_projection_value * threshold_fraction

    # Find segment start and end points based on values below the threshold
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


#read the images taken as input from front end   
def read_input(answer_data, marking_data): 
    nparr_answer = np.frombuffer(answer_data, np.uint8)
    answer_img = cv2.imdecode(nparr_answer, cv2.IMREAD_COLOR)
    nparr_marking = np.frombuffer(marking_data, np.uint8)
    marking_img = cv2.imdecode(nparr_marking, cv2.IMREAD_COLOR)

    answered_image = encode_images(answer_img)
    marking_image = encode_images(marking_img)

    # Get segments
    answer_segments = segment_image(answer_img)
    marking_segments = segment_image(marking_img)
    encoded_answer_segments = [encode_images(segment) for segment in answer_segments]

    return (
        answered_image,
        marking_image,
        answer_segments,
        marking_segments,
        encoded_answer_segments
    )

def process_segments(answer_segments, marking_segments):
    marks = 0
    results = []
    for answer_segment, marking_segment in zip(answer_segments, marking_segments):
        # Extract text from segments
        answer_text = read_text_from_image(answer_segment)
        marking_text = read_text_from_image(marking_segment)

        # Rearrange text using ChatGPT
        rearranged_answer_text = rearrange_text_with_chatgpt(answer_text)
        rearranged_marking_text = rearrange_text_with_chatgpt(marking_text) 
        similarity_result, result_value, marks = compare_similarity(rearranged_marking_text, rearranged_answer_text, marks)
        
        #time.sleep(60)
        results.append((rearranged_answer_text, rearranged_marking_text, similarity_result, result_value, marks))

    return results

#connect to the front end
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        answer = request.files['answer']
        marking = request.files['marking']

        if answer and marking:
            # Read the uploaded image data
            answer_data = answer.read()
            marking_data = marking.read()

            (
                answer_image,
                marking_image,
                answer_segments,
                marking_segments,
                encoded_img
            ) = read_input(answer_data, marking_data)

            results = process_segments(answer_segments, marking_segments)
            
            return render_template(
                'index.html', 
                answer_src=f"data:image/jpeg;base64,{answer_image}",
                marking_src=f"data:image/jpeg;base64,{marking_image}",
                results=results,
                answer_segments_src=[f"data:image/jpeg;base64,{segment}" for segment in encoded_img]
            )

    return render_template('index.html', answer_src=None, marking_src=None, results=None, answer_segments_src=None)
if __name__ == '__main__':
    app.run(debug=True)

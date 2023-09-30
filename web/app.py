from flask import Flask, render_template, request
import base64
import cv2
import numpy as np

app = Flask(__name__)

def read_images(answer_data, marking_data):
    # Read images
    nparr_answer = np.frombuffer(answer_data, np.uint8)
    answer_img = cv2.imdecode(nparr_answer, cv2.IMREAD_COLOR)
    nparr_marking = np.frombuffer(marking_data, np.uint8)
    marking_img = cv2.imdecode(nparr_marking, cv2.IMREAD_COLOR)

    _, encoded_answer = cv2.imencode('.jpg', answer_img)
    encoded_answer = base64.b64encode(encoded_answer).decode('utf-8')
    _, encoded_marking = cv2.imencode('.jpg', marking_img)
    encoded_marking = base64.b64encode(encoded_marking).decode('utf-8')

    # Convert the answer image to grayscale
    gray_answer = cv2.cvtColor(answer_img, cv2.COLOR_BGR2GRAY)
    _, thresholded_answer = cv2.threshold(gray_answer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_answer = cv2.bitwise_not(thresholded_answer)

    # Encode the answer grayscale image to base64
    _, encoded_answer_modified = cv2.imencode('.jpg', inverted_answer)
    encoded_answer_modified = base64.b64encode(encoded_answer_modified).decode('utf-8')

    # Convert the marking image to grayscale
    gray_marking = cv2.cvtColor(marking_img, cv2.COLOR_BGR2GRAY)
    _, thresholded_marking = cv2.threshold(gray_marking, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_marking = cv2.bitwise_not(thresholded_marking)

    # Encode the marking grayscale image to base64
    _, encoded_marking_modified = cv2.imencode('.jpg', inverted_marking)
    encoded_marking_modified = base64.b64encode(encoded_marking_modified).decode('utf-8')

    return f"data:image/jpeg;base64,{encoded_answer}", f"data:image/jpeg;base64,{encoded_marking}", f"data:image/jpeg;base64,{encoded_answer_modified}", f"data:image/jpeg;base64,{encoded_marking_modified}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        answer = request.files['answer']
        marking = request.files['marking']

        if answer and marking:
            # Read the uploaded image data
            answer_data = answer.read()
            marking_data = marking.read()

            answer_image, marking_image, answer_modified, marking_modified = read_images(answer_data, marking_data)

            # Display the resized images on the page
            return render_template(
                'index.html', 
                answer_src=answer_image,
                marking_src=marking_image, 
                answer_modified_src=answer_modified,
                marking_modified_src=marking_modified
            )

    return render_template('index.html', answer_src=None, marking_src=None, answer_modified_src=None, marking_modified_src=None)

if __name__ == '__main__':
    app.run(debug=True)

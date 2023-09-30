from flask import Flask, render_template, request
import base64
import cv2
import numpy as np

app = Flask(__name__)

def read_image(image_data):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresholded)

        # Encode the grayscale image to base64
        _, encoded_image = cv2.imencode('.jpg', inverted)
        encoded_image = base64.b64encode(encoded_image).decode('utf-8') 

        return f"data:image/jpeg;base64,{encoded_image}"

    except Exception as e:
        # Handle decoding or encoding errors
        print(f"Error processing image: {str(e)}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']

        if image:
            # Read the uploaded image data
            image_data = image.read()

            binary_image = read_image(image_data)

            # Display the resized image on the page
            return render_template('index.html', image_src=binary_image)

    return render_template('index.html', image_src=None)

if __name__ == '__main__':
    app.run(debug=True)





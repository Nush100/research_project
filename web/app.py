from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
import openai
from PIL import Image

openai.api_key = ''

tick_image_path = "C:/Study/NSBM/research/sourcecode/images/questions/tick.png"
cross_image_path = "C:/Study/NSBM/research/sourcecode/images/questions/cross.png"

# Read tick and cross images
tick_image = cv2.imread(tick_image_path)
cross_image = cv2.imread(cross_image_path)

app = Flask(__name__)

def evaluate_answer(answer_text):
    prompt = f"Fact check: \"{answer_text}\"\n\nEvaluation:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    # Get the generated response text
    generated_text = response.choices[0].text.strip() 
    if "true" in generated_text.lower():
       sign = tick_image
    else:
        sign = cross_image

    return generated_text, sign


def overlay_tick_on_answer(answer_img, tick_img):
    try:
        # Convert NumPy array to PIL Image
        answer_img_pil = Image.fromarray(cv2.cvtColor(answer_img, cv2.COLOR_BGR2RGB))
        tick_img_pil = Image.fromarray(cv2.cvtColor(tick_img, cv2.COLOR_BGR2RGB))

        # Resize the tick image
        tick_img_pil = tick_img_pil.resize((80, 80), resample=Image.BICUBIC)

        # Ensure both images have an alpha channel
        answer_img_pil = answer_img_pil.convert('RGBA')
        tick_img_pil = tick_img_pil.convert('RGBA')

        # Calculate the position to overlay the tick image in the middle of the answer image
        x_position = (answer_img_pil.width - tick_img_pil.width) // 2
        y_position = (answer_img_pil.height - tick_img_pil.height) // 2

        # Create a composite image with transparency
        composite = Image.alpha_composite(answer_img_pil, Image.new('RGBA', answer_img_pil.size, (0, 0, 0, 0)))
        composite.paste(tick_img_pil, (x_position, y_position), tick_img_pil)

        # Convert the result back to NumPy array
        result_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGBA2BGR)
        return result_img
    except Exception as e:
        print(f"Error in overlay_tick_on_answer: {str(e)}")
        return answer_img
    
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

    # Evaluate the answer using ChatGPT
    evaluation, sign = evaluate_answer("Fedrick North is the First Bristish Governer in Sri Lanka")
    answer = overlay_tick_on_answer(answer_img, sign)
    
    # Encode the answer grayscale image to base64
    _, encoded_answer_modified = cv2.imencode('.jpg', answer)
    encoded_answer_modified = base64.b64encode(encoded_answer_modified).decode('utf-8')

    return (
        f"data:image/jpeg;base64,{encoded_answer}",
        f"data:image/jpeg;base64,{encoded_marking}",
        f"data:image/jpeg;base64,{encoded_answer_modified}",
        evaluation
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        answer = request.files['answer']
        marking = request.files['marking']

        if answer and marking:
            # Read the uploaded image data
            answer_data = answer.read()
            marking_data = marking.read()

            answer_image, marking_image, answer_modified, evaluation_result = read_images(answer_data, marking_data)

            # Display the resized images on the page
            return render_template(
                'index.html', 
                answer_src=answer_image,
                marking_src=marking_image, 
                answer_modified_src=answer_modified,
                evaluation_result=evaluation_result
            )

    return render_template('index.html', answer_src=None, marking_src=None, answer_modified_src=None, evaluation_result='')

if __name__ == '__main__':
    app.run(debug=True)

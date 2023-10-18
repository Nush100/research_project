from flask import Blueprint, render_template, request
from website.common import encode_images, read_input 

marking =  Blueprint("marking", __name__)


@marking.route('/ms', methods=['GET', 'POST'])
def marking_with_marking_scheme():
    if request.method == 'POST':
        answer = request.files['answer']
        marking = request.files['marking']

        if answer and marking:
            # Read the uploaded image data
            answer_data = answer.read()
            marking_data = marking.read()
            
            answer_img = read_input(answer_data)
            marking_img = read_input(marking_data)
            
            answered_image = encode_images(answer_img)
            marking_image = encode_images(marking_img)
            
            return render_template(
                'marking_scheme.html', 
                answer_src=f"data:image/jpeg;base64,{answered_image}",
                marking_src=f"data:image/jpeg;base64,{marking_image}"
            )

    return render_template('marking_scheme.html', answer_src=None, marking_src=None)


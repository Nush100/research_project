from flask import Blueprint, render_template, request
from website.common import read_input, segment_image, text_in_segment, remove_digits_and_symbols, check_correctness_with_openai, encode_images, combine_color_images_vertically, add_text_to_binary_image, create_text_image

marking_ai =  Blueprint("marking_ai", __name__)


@marking_ai.route('/ai', methods=['GET', 'POST'])
def marking_with_ai():
    if request.method == 'POST':
        answer = request.files['answer']
    
        if answer:
            results = []
            marks = 0
            answer_data = answer.read() 
            
            answer_img = read_input(answer_data) 
            segments = segment_image(answer_img)

            for segment in segments:
                text, confidence = text_in_segment(segment)
                cleaned_text = remove_digits_and_symbols(text)   
                correctness, correct_value, marks = check_correctness_with_openai(cleaned_text, marks)
                answer_image = encode_images(segment)
                answer_segment=f"data:image/jpeg;base64,{answer_image}" 
                answer_with_text = add_text_to_binary_image(segment, correct_value)
                results.append((text, confidence, cleaned_text, correctness, correct_value, marks, answer_segment, answer_with_text))

            marks_list = [result[5] for result in results]
            color_images_list = [result[7] for result in results]
            text_image = create_text_image(f"Total marks {marks_list[-1]}")
            color_images_list.append(text_image)
            combined_image = combine_color_images_vertically(color_images_list)
            encoded_combine = encode_images(combined_image)
            answered_image = encode_images(answer_img) 

            return render_template(
                'marking_ai.html', 
                answer_src=f"data:image/jpeg;base64,{answered_image}",
                results=results,
                combined_image=f"data:image/jpeg;base64,{encoded_combine}",
            )

    return render_template('marking_ai.html', answer_src=None, results=None, combined_image=None)
from flask import Blueprint, render_template, request
from website.common import read_input, divide_image, section_heights, segment_image, text_in_segment, binary_to_rgb, remove_digits_and_symbols, check_correctness_with_openai, encode_images, combine_color_images_vertically, add_symbol, create_text_image
import time

marking_ai =  Blueprint("marking_ai", __name__)


@marking_ai.route('/ai', methods=['GET', 'POST'])
def marking_with_ai():
    if request.method == 'POST':
        answer = request.files['answer']
        institution = request.form['institution']
        correct_mark = int(request.form['correct_mark']) 
    
        if answer:
            results = []
            marks = 0
            answer_data = answer.read() 
            
            answer_img = read_input(answer_data) 
            height = section_heights.get(institution)
            upper_part, lower_part = divide_image(answer_img, height)
            segments = segment_image(lower_part)
            print(len(segments))
            for i, segment in enumerate(segments[3:]):
                text, confidence = text_in_segment(segment)
                cleaned_text = remove_digits_and_symbols(text) 
                if (i + 1) % 3 == 0:
                    time.sleep(60)  
                correctness, correct_value, marks = check_correctness_with_openai(cleaned_text, marks, correct_mark) 
                answer_segment = f"data:image/jpeg;base64,{encode_images(segment)}" 
                answer_with_text = add_symbol(segment, correct_value)
                results.append((text, confidence, cleaned_text, correctness, correct_value, marks, answer_segment, answer_with_text))

            marks_list = [result[5] for result in results]
            upper = binary_to_rgb(upper_part)
            color_images_list=[upper, segments[0], segments[1]]
            color_images_list.extend(result[7] for result in results)
            text_image = create_text_image(f"Total {marks_list[-1]}")
            color_images_list.append(text_image)
            combined_image = combine_color_images_vertically(color_images_list)
            encoded_combine = encode_images(combined_image)
            answered_image = encode_images(answer_img) 
            encoded_upper = encode_images(upper_part)
            encoded_lower = encode_images(lower_part)

            return render_template(
                'marking_ai.html',
                answer_src=f"data:image/jpeg;base64,{answered_image}",
                results=results,
                combined_image=f"data:image/jpeg;base64,{encoded_combine}",
                encoded_upper=f"data:image/jpeg;base64,{encoded_upper}",
                encoded_lower=f"data:image/jpeg;base64,{encoded_lower}",
            )

    return render_template('marking_ai.html', answer_src=None, results=None, combined_image=None, encoded_upper=None, encoded_lower=None)
from flask import Blueprint, render_template, request
from website.common import read_input, divide_image, section_heights, are_texts_similar, binary_to_rgb, segment_image, text_in_segment, remove_digits_and_symbols, check_correctness_with_openai, encode_images, combine_color_images_vertically, add_symbol, create_text_image


marking =  Blueprint("marking", __name__)


@marking.route('/ms', methods=['GET', 'POST'])
def marking_with_marking_scheme():
    if request.method == 'POST':
        answer = request.files['answer']
        marking = request.files['marking']
        institution = request.form['institution']
        correct_mark = int(request.form['correct_mark']) 

        if answer and marking:
            results = []
            marks = 0
            # Read the uploaded image data
            answer_data = answer.read()
            marking_data = marking.read()
            
            answer_img = read_input(answer_data)
            marking_img = read_input(marking_data)
            height = section_heights.get(institution)

            upper_part_answer, lower_part_answer = divide_image(answer_img, height)
            upper_part, lower_part = divide_image(marking_img, height)
            segments_answer= segment_image(lower_part_answer)
            segments = segment_image(lower_part)

            for answer_segment, segment in zip(segments_answer[3:8], segments[2:8]):
                text_answer, confidence =  text_in_segment(answer_segment)
                text, _ = text_in_segment(segment)
                cleaned_answer = remove_digits_and_symbols(text_answer)
                cleaned_marking = remove_digits_and_symbols(text)
                similarity, marks, similarity_ratio = are_texts_similar(cleaned_marking, cleaned_answer, correct_mark, marks)
                answer = f"data:image/jpeg;base64,{encode_images(answer_segment)}" 
                answer_with_symbol = add_symbol(answer_segment, similarity)
                results.append((cleaned_answer, confidence, cleaned_marking, similarity, marks, answer, answer_with_symbol, similarity_ratio))

            marks_list =  [result[4] for result in results]
            upper_part_binary = binary_to_rgb(upper_part_answer)
            color_images_list=[upper_part_binary, segments[0], segments[1]]
            answered_image = encode_images(answer_img)
            marking_image = encode_images(marking_img)
            color_images_list.extend(result[6] for result in results)
            text_image = create_text_image(f"Total = {marks_list[-1]}")
            color_images_list.append(text_image)
            combined_image = combine_color_images_vertically(color_images_list)
            encoded_combine = encode_images(combined_image)
            encoded_upper = encode_images(upper_part_answer)
            encoded_lower = encode_images(lower_part_answer)

            return render_template(
                'marking_scheme.html', 
                answer_src=f"data:image/jpeg;base64,{answered_image}",
                marking_src=f"data:image/jpeg;base64,{marking_image}",
                results=results,
                encoded_upper=f"data:image/jpeg;base64,{encoded_upper}",
                encoded_lower=f"data:image/jpeg;base64,{encoded_lower}",
                encoded_combine=f"data:image/jpeg;base64,{encoded_combine}"
            )

    return render_template('marking_scheme.html', answer_src=None, marking_src=None)


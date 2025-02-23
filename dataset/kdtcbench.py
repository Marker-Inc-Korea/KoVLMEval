import torch

from tqdm import tqdm
from PIL import Image
from io import BytesIO

#from llava.mm_utils import tokenizer_image_token, process_images

def check_options(options):
    
    options_prompt = []
    for i in range(len(options)):
        
        idx = options.index[i]
        row = options[i]
        #print(idx, row)
        
        if row is not None:
            prompt = idx + '. ' + row.strip()
            options_prompt.append(prompt)
    
    return options_prompt

def read_image(bytes):
    try:
        image = Image.open(BytesIO(bytes))
        return image
    
    except Exception as e:
        raise Exception(e)


def kdtcbench_eval(eval_dataset, 
                  model,
                  text_tokenizer, 
                  visual_tokenizer,
                  category):
    
    correct_count = 0
    unknown_count = 0
    unknown_list = []
    
    if category == 'ovis':
        for i in tqdm(range(len(eval_dataset))):
            
            #print(eval_dataset.columns)
            
            question = eval_dataset['question'][i]
            choice_a = eval_dataset['choice_a'][i]
            choice_b = eval_dataset['choice_b'][i]
            choice_c = eval_dataset['choice_c'][i]
            choice_d = eval_dataset['choice_d'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            ground_truth = eval_dataset['answer'][i]
            
            # query
            query = f'<image>\n{question}\nOptions: A: {choice_a}, B: {choice_b}, C: {choice_c}, D: {choice_d}\n주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.'
            #print(query)
            
            # format conversation
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
            
            # generate output
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f'Output:\n{output}')
                
            pred_value = output.strip()[0]
            
            # post-processing
            if ground_truth == pred_value:
                correct_count += 1
                print(correct_count)
            else:
                duplicated_check = 0
                if 'A' in output:
                    pred_value = 'A'
                    duplicated_check += 1
                elif 'B' in output:
                    pred_value = 'B'
                    duplicated_check += 1
                elif 'C' in output:
                    pred_value = 'C'
                    duplicated_check += 1
                elif 'D' in output:
                    pred_value = 'D'
                    duplicated_check += 1
                
                if (duplicated_check==1) and (ground_truth == pred_value):
                    correct_count += 1
                    print(correct_count)
        
    elif category == 'bllossom':
        for i in tqdm(range(len(eval_dataset))):
            
            question = eval_dataset['question'][i]
            choice_a = eval_dataset['choice_a'][i]
            choice_b = eval_dataset['choice_b'][i]
            choice_c = eval_dataset['choice_c'][i]
            choice_d = eval_dataset['choice_d'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            ground_truth = eval_dataset['answer'][i]

            instruction = f'{question}\nOptions: A: {choice_a}, B: {choice_b}, C: {choice_c}, D: {choice_d}\n주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.'
            messages = [
                {'role': 'user','content': [
                    {'type':'image'},
                    {'type': 'text','text': instruction}
                    ]},
                ]
            
            input_text = text_tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
            #print(input_text)
            
            inputs = text_tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            
            # generate output
            output = model.generate(**inputs, 
                                    max_new_tokens=1024,
                                    temperature=None,
                                    eos_token_id=text_tokenizer.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                                    use_cache=True) # If False, 60 hours
            #print(text_tokenizer.decode(output[0]))
            
            output = text_tokenizer.decode(output[0])[len(input_text):].strip()
            print(f'Output:\n{output}')
            pred_value = output.strip()[0]
            
            # post-processing
            if ground_truth == pred_value:
                correct_count += 1
                print(correct_count)
            else:
                duplicated_check = 0
                if 'A' in output:
                    pred_value = 'A'
                    duplicated_check += 1
                elif 'B' in output:
                    pred_value = 'B'
                    duplicated_check += 1
                elif 'C' in output:
                    pred_value = 'C'
                    duplicated_check += 1
                elif 'D' in output:
                    pred_value = 'D'
                    duplicated_check += 1
                
                if (duplicated_check==1) and (ground_truth == pred_value):
                    correct_count += 1
                    print(correct_count)
    
    elif category == 'VARCO':
        error_count = 0
        error_list = []
        
        for i in tqdm(range(len(eval_dataset))):
        
            question = eval_dataset['question'][i]
            choice_a = eval_dataset['choice_a'][i]
            choice_b = eval_dataset['choice_b'][i]
            choice_c = eval_dataset['choice_c'][i]
            choice_d = eval_dataset['choice_d'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            ground_truth = eval_dataset['answer'][i]
            
            instruction = f'{question}\nOptions: A: {choice_a}, B: {choice_b}, C: {choice_c}, D: {choice_d}\n주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.'
            query = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image"},
                    ],
                },
            ]
        
            #try:
            # preprocessing
            prompt = text_tokenizer.apply_chat_template(query, add_generation_prompt=True)
            #print(prompt)
            
            EOS_TOKEN = "<|im_end|>"
            inputs = text_tokenizer(images=image, text=prompt, return_tensors='pt').to(torch.float16).to(model.device)
            #print(inputs)
            
            # generation
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = text_tokenizer.batch_decode(output_ids[0][inputs.input_ids.shape[1]:])
            outputs = ''.join(outputs).strip()
            if outputs.endswith(EOS_TOKEN):
                outputs = outputs[: -len(EOS_TOKEN)]
            print(f'Output:\n{outputs}')
            pred_value = outputs.strip()[0]
            
            # post-processing
            if ground_truth == pred_value:
                correct_count += 1
                print(correct_count)
            else:
                duplicated_check = 0
                if 'A' in outputs:
                    pred_value = 'A'
                    duplicated_check += 1
                elif 'B' in outputs:
                    pred_value = 'B'
                    duplicated_check += 1
                elif 'C' in outputs:
                    pred_value = 'C'
                    duplicated_check += 1
                elif 'D' in outputs:
                    pred_value = 'D'
                    duplicated_check += 1
                
                if (duplicated_check==1) and (ground_truth == pred_value):
                    correct_count += 1
                    print(correct_count)

            #except Exception as E:
            #    error_count += 1
            #    error_list.append(prompt)
            #    #print(E)
        
    return correct_count/len(eval_dataset)
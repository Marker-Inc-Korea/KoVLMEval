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


def kllavaw_eval(eval_dataset, 
                  model,
                  text_tokenizer, 
                  visual_tokenizer,
                  category,
                  gpt_client):
    
    score = 0
    
    if category == 'ovis':
        for i in tqdm(range(len(eval_dataset))):
            
            #print(eval_dataset.columns)
            
            question = eval_dataset['question'][i]
            caption = eval_dataset['caption'][i]
            gpt_answer = eval_dataset['gpt_answer'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            # query
            query = f'<image>\n{question}'
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
                #print(f'Output:\n{output}')
                
            target_model_answer = output
            
            ## Evaution prompt
            gpt_eval_prompt = f'''[설명]
{caption}

[질문]
{question}

[어시스턴트 1]
{gpt_answer}
[어시스턴트 1 끝]

[어시스턴트 2]
{target_model_answer}
[어시스턴트 2 끝]

[System]
두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.

# 단계
1. 제공된 이미지 [설명]을 검토하세요.
2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:
- `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?
- `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?
- `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?
- `세부 수준`: 응답이 과하지 않게 충분히 자세한가?
- `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?
3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.
4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.
5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.

# 출력 형식
- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)
- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.

# 주의사항
- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.
- 분석과 설명에서 일관성과 명확성을 유지하세요.'''
            #print(gpt_eval_prompt)
            
            ## Evaluation
            chat_completion = gpt_client.chat.completions.create(
                messages=[
                    {
                        'role':'user',
                        'content':gpt_eval_prompt,
                    }
                ],
                model='gpt-4o-2024-08-06' # == gpt-4o
            )
            content = chat_completion.choices[0].message.content.strip()
            print(content)
            if content[:2] == '10':
                gpt_score = 10.0
                if content[3:5] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[3])
            else:
                gpt_score = float(content[0])
                if content[2:4] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[2])
            
            score = score + model_score
            print("Score:", model_score)
                
    elif category == 'bllossom':
        for i in tqdm(range(len(eval_dataset))):
            
            question = eval_dataset['question'][i]
            caption = eval_dataset['caption'][i]
            gpt_answer = eval_dataset['gpt_answer'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            # query
            query = f'{question}'
            messages = [
                {'role': 'user','content': [
                    {'type':'image'},
                    {'type': 'text','text': query}
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
            #print(f'Output:\n{output}')
            
            target_model_answer = output
            
            ## Evaution prompt
            gpt_eval_prompt = f'''[설명]
{caption}

[질문]
{question}

[어시스턴트 1]
{gpt_answer}
[어시스턴트 1 끝]

[어시스턴트 2]
{target_model_answer}
[어시스턴트 2 끝]

[System]
두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.

# 단계
1. 제공된 이미지 [설명]을 검토하세요.
2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:
- `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?
- `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?
- `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?
- `세부 수준`: 응답이 과하지 않게 충분히 자세한가?
- `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?
3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.
4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.
5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.

# 출력 형식
- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)
- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.

# 주의사항
- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.
- 분석과 설명에서 일관성과 명확성을 유지하세요.'''
            #print(gpt_eval_prompt)
            
            ## Evaluation
            chat_completion = gpt_client.chat.completions.create(
                messages=[
                    {
                        'role':'user',
                        'content':gpt_eval_prompt,
                    }
                ],
                model='gpt-4o-2024-08-06' # == gpt-4o
            )
            content = chat_completion.choices[0].message.content.strip()
            print(content)
            if content[:2] == '10':
                gpt_score = 10.0
                if content[3:5] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[3])
            else:
                gpt_score = float(content[0])
                if content[2:4] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[2])
            
            score = score + model_score
            print("Score:", model_score)
    
    elif category == 'VARCO':
        error_count = 0
        error_list = []
        
        for i in tqdm(range(len(eval_dataset))):
        
            question = eval_dataset['question'][i]
            caption = eval_dataset['caption'][i]
            gpt_answer = eval_dataset['gpt_answer'][i]
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
            #print(image)
            
            # query
            instruction = f'{question}'
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
            #print(f'Output:\n{output}')
                
            target_model_answer = outputs
            
            ## Evaution prompt
            gpt_eval_prompt = f'''[설명]
{caption}

[질문]
{question}

[어시스턴트 1]
{gpt_answer}
[어시스턴트 1 끝]

[어시스턴트 2]
{target_model_answer}
[어시스턴트 2 끝]

[System]
두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.

# 단계
1. 제공된 이미지 [설명]을 검토하세요.
2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:
- `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?
- `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?
- `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?
- `세부 수준`: 응답이 과하지 않게 충분히 자세한가?
- `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?
3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.
4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.
5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.

# 출력 형식
- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)
- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.

# 주의사항
- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.
- 분석과 설명에서 일관성과 명확성을 유지하세요.'''
            #print(gpt_eval_prompt)
            
            ## Evaluation
            chat_completion = gpt_client.chat.completions.create(
                messages=[
                    {
                        'role':'user',
                        'content':gpt_eval_prompt,
                    }
                ],
                model='gpt-4o-2024-08-06' # == gpt-4o
            )
            content = chat_completion.choices[0].message.content.strip()
            print(content)
            if content[:2] == '10':
                gpt_score = 10.0
                if content[3:5] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[3])
            else:
                gpt_score = float(content[0])
                if content[2:4] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[2])
                
            score = score + model_score
            print("Score:", model_score)

            #except Exception as E:
            #    error_count += 1
            #    error_list.append(prompt)
            #    #print(E)
            #break
            
    elif category == 'GPT':
        
        import base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
        for i in tqdm(range(len(eval_dataset))):
            
            question = eval_dataset['question'][i]
            caption = eval_dataset['caption'][i]
            gpt_answer = eval_dataset['gpt_answer'][i]
            #print(eval_dataset.columns)
            
            try:
                img_bytes = eval_dataset['image'][i]['bytes']
                image = read_image(img_bytes)
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                img_bytes = eval_dataset['image.bytes'][i]
                image = read_image(img_bytes)
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            #print(image)
            #print(img_base64)
            
            # query
            query = f'{question}'
            chat_completion = gpt_client.chat.completions.create(
                messages=[
                    {
                        'role':'user',
                        'content':[
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ],
                    }
                ],
                model=model, # == gpt-4o
                max_tokens=1024
            )
            
            target_model_answer = chat_completion.choices[0].message.content.strip()
            
            ## Evaution prompt
            gpt_eval_prompt = f'''[설명]
{caption}

[질문]
{question}

[어시스턴트 1]
{gpt_answer}
[어시스턴트 1 끝]

[어시스턴트 2]
{target_model_answer}
[어시스턴트 2 끝]

[System]
두 인공지능 어시스턴트의 성능을 [질문]에 대한 응답에 기반하여 평가하세요. 해당 [질문]은 특정 이미지를 보고 생성되었습니다. `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력`을 기준으로 응답을 평가하세요. 각각의 어시스턴트에게 1에서 10까지의 전반적인 점수를 부여하며, 높은 점수일수록 더 나은 전반적인 성능을 나타냅니다.

# 단계
1. 제공된 이미지 [설명]을 검토하세요.
2. 각 어시스턴트의 응답을 다음 기준으로 분석하세요:
- `유용성`: 응답이 사용자의 질문을 얼마나 잘 해결하는가?
- `관련성`: 응답이 사용자의 질문에 얼마나 적절한가?
- `정확성`: 응답에서 제공한 정보가 얼마나 정확한가?
- `세부 수준`: 응답이 과하지 않게 충분히 자세한가?
- `한국어 생성능력`: 생성된 한국어 문장이 자연스럽고 문법적으로 올바른가?
3. 분석에 기반하여 각 어시스턴트에게 1에서 10까지의 점수를 부여하세요.
4. 두 점수를 공백으로 구분하여 한 줄로 제공하세요.
5. 점수에 대한 이유를 강조하면서 포괄적인 평가를 제공하고, 편견을 피하며 응답의 순서가 판단에 영향을 미치지 않도록 하세요.

# 출력 형식
- 첫 번째 줄: `어시스턴트1_점수 어시스턴트2_점수` (예: `8 9`)
- 두 번째 줄: `유용성`, `관련성`, `정확성`, `세부 수준`, `한국어 생성능력` 기준으로 점수를 설명하는 자세한 문단을 제공합니다.

# 주의사항
- 평가 시 잠재적 편견을 방지하여 객관성을 확보하세요.
- 분석과 설명에서 일관성과 명확성을 유지하세요.'''
            #print(gpt_eval_prompt)
            
            ## Evaluation
            chat_completion = gpt_client.chat.completions.create(
                messages=[
                    {
                        'role':'user',
                        'content':gpt_eval_prompt,
                    }
                ],
                model='gpt-4o-2024-08-06' # == gpt-4o
            )
            content = chat_completion.choices[0].message.content.strip()
            print(content)
            if content[:2] == '10':
                gpt_score = 10.0
                if content[3:5] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[3])
            else:
                gpt_score = float(content[0])
                if content[2:4] == '10':
                    model_score = 10.0
                else:
                    model_score = float(content[2])
            
            score = score + model_score
            print("Score:", model_score)
        
    return score/len(eval_dataset)
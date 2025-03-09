import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

#use the model we fine tuned
model_path = './motivational_model'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

#set title of webpage
st.title("Daily dose of motivation!")

#Define moods to be in the radio menu
moods = ['Courage', 'Perseverance', 'Goal Setting', 'Confidence', 'Growth Mindset', 'Resilience', 'Focus', 'Teamwork']
selected = st.radio('What do you need today?', moods)
user_input = st.text_area('Generate a motivational quote about...', f"Focus: {selected}")

#generate motivational quote based on input. generate a warning if no user input is entered
if st.button("Generate Text"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors='pt')
        outputs = model.generate(input_ids = inputs['input_ids'], 
                            attention_mask = inputs['attention_mask'],
                            max_length = 35, 
                            min_length = 10,
                            do_sample=True, 
                            temperature=0.9, 
                            no_repeat_ngram_size=2, 
                            top_k=80, 
                            num_beams=2,
                            top_p = 0.92,
                            repetition_penalty=1.3, 
                            pad_token_id = tokenizer.eos_token_id,
                            eos_token_id = tokenizer.eos_token_id,
                            early_stopping = True)
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().capitalize().split('quote:', 1)[-1].strip()
        if not gen_text.endswith('.'):
            gen_text += '.'
        st.markdown(gen_text)
    else:
        st.warning('Please enter a prompt')
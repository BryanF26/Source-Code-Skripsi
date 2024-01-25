import streamlit as st
import requests
import pandas as pd
from fastapi import FastAPI
from happytransformer import HappyGeneration, GENSettings
import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
import json
f = open('Data.json')
data_product = json.load(f)
f.close()

data_store = []

def settings(min_val= 50, max_val=100, temp=0.7, top_ks=50):
    min_length =  min_val
    max_length = max_val
    do_sample = True
    early_stopping = True
    num_beams = 1 
    temperature = temp
    top_k = top_ks
    top_p = temp
    no_repeat_ngram_size = 2
    gen_args = GENSettings(min_length, max_length, do_sample, early_stopping, num_beams, temperature, top_k, no_repeat_ngram_size, top_p)
    return gen_args

def calculate_rouge_l_f_scores(model_summaries, reference_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(model_summaries, reference_summaries)
    return scores[0]["rouge-l"]["f"]

def calculate_bleu_scores(models, references):
    model_tokens = nltk.word_tokenize(models)
    reference_tokens = [nltk.word_tokenize(references)]
    bleu_scores = sentence_bleu(reference_tokens, model_tokens)
    return bleu_scores

def get_data(category, description):
    for i in data_product[category]:
        if i['description'] == description:
            return i

def generate_gpt2(category: str, description: str, min_val: int, max_val: int, temps: float, top_ks: int):
    product = get_data(category, description)
    happy_gen = HappyGeneration(load_path="./GPT 2")
    gen_args = settings(min_val, max_val, temps, top_ks)
    descs = product['description'].split(' ')
    text = f"""Categories: {category}
    Title: {product['title']}
    Features: {product['feature']}
    Description: {descs[0]} {descs[1]}"""
    result = happy_gen.generate_text(text, args=gen_args)
    output = result.text
    try:
        output = output.split('Description:')[1]
    except:
        output = output
    bleu_score = calculate_bleu_scores(output, description)
    rouge_score = calculate_rouge_l_f_scores(output, description)
    texts = [output, bleu_score, rouge_score]
    return texts

def generate_gptneo(category: str, description: str, min_val: int, max_val: int, temps: float, top_ks: int):
    product = get_data(category, description)
    happy_gen = HappyGeneration(load_path="./GPT Neo")
    gen_args = settings(min_val, max_val, temps, top_ks)
    descs = product['description'].split(' ')
    text = f"""Categories: {category}
    Title: {product['title']}
    Features: {product['feature']}
    Description: {descs[0]} {descs[1]}"""
    result = happy_gen.generate_text(text, args=gen_args)
    output = result.text
    try:
        output = output.split('Description:')[1]
    except:
        output = output
    bleu_scoree = calculate_bleu_scores(output, description)
    rouge_score = calculate_rouge_l_f_scores(output, description)
    texts = [output, bleu_scoree, rouge_score]
    return texts

def generate(category: str, title: str, feature: str):
    happy_gen = HappyGeneration(load_path="./GPT Neo")
    gen_args = settings()
    text = f"""Categories: {category}
    Title: {title}
    Features: {feature}
    Description: """
    result = happy_gen.generate_text(text, args=gen_args)
    output = result.text
    try:
        output = output.split('Description:')[1]
    except:
        output = output
    data = [category, title, feature, output]
    data_store.append(data)
    return output

def show_description():
    return data_product

def show_history():
    return data_store

def delete_all_history():
    data_store.clear()

def test_model_button():
    if st.button('Test Model'):
        if selected_model == "GPT 2":
            score = generate_gpt2(selected_category, selected_description, min_val, max_val, temps, top_ks)
        if selected_model == "GPT Neo":
            score = generate_gptneo(selected_category, selected_description, min_val, max_val, temps, top_ks)
        evaluation = score
        st.write(f'Generated Description: {evaluation[0]}')
        st.write(f'Generated Bleu Score: {evaluation[1]}')
        st.write(f'Generated Rouge-L Score: {evaluation[2]}')
        st.success('You have success test model')

def generated_text_button():
    if st.button('Generate Text'):
        if title_input == "" or feature_input == "":
            return st.warning('Please fill Title and Features')
        generated_text = generate(selected_category, title_input, feature_input)
        st.write(f'Generated Description: {generated_text}')
        st.success('You have success generate product description')

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Test Model", "Generate","History"])

with tab1:
    st.title('Introduction')
    st.write("Welcome to our website Product Description Generator. Using sophisticated language models, GPT-2 and GPT-Neo, we provide a breakthrough tool for creating high-quality, varied, and interesting writing.")
    st.write("We use GPT-2, an advanced language model created by OpenAI that is well-known for its capacity to generate coherent and contextually appropriate text in response to a given prompt. In addition, we introduce GPT-Neo, an open-source alternative that matches the GPT-3 design.")
    st.write("There are three main menu that you can explore. First is Test model which you can test the model GPT-2 or GPT-Neo and choose description which has been provided using your own parameter. Second is Generate to generate product description with your input title and feature using our proposed model GPT-Neo. Lastly, you can find your generated text in History.") 

with tab2:
    st.title('Testing Model')
    
    models = ['GPT 2', 'GPT Neo']
    selected_model = st.selectbox('Model:', models, key="model")
    
    categories = ['Electronics', 'Home & Kitchen', 'Toys & Games']
    selected_category = st.selectbox('Category:', categories, key="category")
    
    description = show_description()
    selected_description = st.selectbox('Description:', options=[d["description"] for d in description[selected_category]], key="description")
    
    min_val = st.slider('Min (Minimal Length Generate)', min_value=10, max_value=100, value=10)
    max_val = st.slider('Max (Maximal Length Generate)', min_value=10, max_value=100, value=50)
    temps = st.slider('Temperature (Unique Word Generate)', min_value=0.1, max_value=1.0, value=0.7)
    top_ks = st.slider('Top_K (Possiblity Word Generate)', min_value=0, max_value=100, value=50)
    
    test_model_button()
    
    
with tab3:
    st.expander('Generate',False)
    st.title('Text Generation')
    categories = ['Electronics', 'Home & Kitchen', 'Toys & Games']
    selected_category = st.selectbox('Category:', categories)
    title_input = st.text_input('Title:')
    feature_input = st.text_area('Feature:')
    generated_text_button()

with tab4:
    st.title('History')
    data = []
    data = show_history()
    df = pd.DataFrame(data,columns=['Category', 'Title', 'Feature', 'Description'])
    x = st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button('Clear History'):
        delete_all_history()
        st.success('You have success delete history')
        
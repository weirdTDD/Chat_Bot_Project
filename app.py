import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os 
import streamlit as st


st.set_page_config(page_title = "Support Chat", page_icon = "💭", layout = "wide")

@st.cache_resource
def load_system():
    try:
        required_files = ['knowledge_base.pkl', 'ecommerce_index.faiss', 'model_name.txt']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f'Missing required file:{file}')
            return None, None, None
       
        with open('model_name.txt', 'r') as f:
            model_name = f.read().strip()

        with open('knowledge_base.pkl', 'rb') as f:
            knowledge_base = pickle.load(f)

        index = faiss.read_index('ecommerce_index.faiss')

        return model_name, knowledge_base, index 
    
    except Exception as e:
        st.error(f'Error loading system: {str(e)}')
        return None, None, None 
    

def get_answer(query, model_name, knowledge_base, index):
    try:
        query_embedding = model_name.encode([query])
        faiss.normalize_L2(query_embedding)

        score, indices = index.search(query_embedding.astype('float32'), 3)
        best_idk = indices[0][0]
        best_score = scores[0][0]

        if best_score < 0.3:
            return get_fallback(query)
        
        best_match = knowledge_base[best_idk]

        return{
            'answer': best_match['answer'],
            'confidence': "High"if best_score >0.7 else "Medium"
        }
    
    except Exception as e:
        return {'answer': f'Sorry, an error occurred: {str(e)}', 'confidence': 'Low'}
    

def get_fallback(query):
    query_lower = query.lower()

    responses ={
        'track': "Track your order in 'My account' > 'Order History'",
        'return':"We offer 30-day returns. Start in your account.",
        'refund':"Refunds process takes in 5-7 business days.",
        'cancel':"Cancel orders within 1 in your account",
        'shipping':"Standard: 3-5 days, Express: 1-2 days.",
        'payment':"we accept cards, Paypal, Apple pay, and Mobile money payments.",
    }

    for keyword, response in responses.items():
        if keyword in query_lower:
            return {'answer': response, 'confidence': 'Medium'} 
        
    return{
        'answer': "I'm here to help! Ask about orders, shipping, returns, or payments.",
        'confidence': 'Low'
    }


def main():
    st.title("💭 Support Chat")

    model_name, knowledge_base, index = load_system()
    
    if not all([model_name, knowledge_base, index]):
        st.stop()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    #Display chat
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    #Display Input
    for prompt := st.chat_input("How can I help you?..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            response = get_answer(prompt, model_name, knowledge_base, index)
            st.write(response['answer'])

            st.session_state.messages.append({
                'role': 'assistant', 
                'content': response['answer']
            })

        

if __name__ == "__main__":
    main()
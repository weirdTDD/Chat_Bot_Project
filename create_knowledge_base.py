import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os 


def create_ecommerce_knowledge_base():
    #Create a vector database from Bitext e-commerce dataset

    print("Loading Bitext e-commerce dataset...")

    #Load the dataset
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")


    print(f"Dataset loaded: {len(dataset['train'])} examples")

    #prepare knowledge base
    knowledge_base = []
    for example in dataset['train']:

        #clean the response 
        response = example['response']
        response = response.replace("{{Order Number}}", "your order")
        response = response.replace("{{Online Company Portal Info}}", "our website")
        response = response.replace("{{Online Order Interaction}}", "order history")
        response = response.replace("{{Customer Support Hours}}", "business hours")
        response = response.replace("{{Customer Support Phone Number}}", "our support line")
        response = response.replace("{{Website URL}}", "our website")

        knowledge_base.append({
            'question': example['instruction'],
            'answer': response,
            'intent': example['intent'],
            'category': example['category'],
        })

    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 

    #Create embeddings for all questions
    print("Creating embeddings...")
    questions = [item['question'] for item in knowledge_base]
    embeddings = model.encode(questions, show_progress_bar=True)
    print("Embeddings created.")

    #Create FAISS Index
    print("Creating FAISS index...")
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimensions)

    #Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    #Save everything + knowledge base
    print("Saving knowledge base and index...")

    with open('knowledge_base.pkl', 'wb') as f:
        pickle.dump(knowledge_base, f)

    #Save FAISS index
    faiss.write_index(index, 'ecommerce_index.faiss')
    with open('model_name.txt', 'w') as f:
        f.write('sentence-transformers/all-MiniLM-L6-v2')

    print("Knowledge base and index created successfully!")
    print("Files created:")
    print("- knowledge_base.pkl")
    print("- ecommerce_index.faiss")
    print("- model_name.txt")



if __name__ == "__main__":
    create_ecommerce_knowledge_base()
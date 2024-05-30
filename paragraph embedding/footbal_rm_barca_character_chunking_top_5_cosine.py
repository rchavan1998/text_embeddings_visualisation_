import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
from vertexai import init
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Init vertex ai
project_id = "miracleinternproj1"
location = "us-central1"

init(project=project_id, location=location)  
model_name = "text-embedding-004"
model = TextEmbeddingModel.from_pretrained(model_name)

def get_chunks(text, chunk_size=500):
    """Divide text into chunks of specified size."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings(texts, task_type="QUESTION_ANSWERING"):
    """Embeds a list of texts."""
    inputs = [TextEmbeddingInput(text, task_type) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

def fetch_and_process_article(title):
    """Fetch , chunk and calculate vectors for embeddings."""
    page = wikipedia.page(title)
    chunks = get_chunks(page.content)
    embeddings = get_embeddings(chunks)
    return chunks, embeddings

def store_embeddings(filename, titles, chunks, embeddings):
    """Store chunks and embeddings in a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        for title, chunk, embedding in zip(titles, chunks, embeddings):
            writer.writerow([title, chunk] + list(embedding))

def load_embeddings(filename):
    """Load embeddings and chunks from a CSV file."""
    df = pd.read_csv(filename, header=None, quotechar='"', engine='python', on_bad_lines='skip')
    titles = df[0].tolist()
    chunks = df[1].tolist()
    embeddings = df.iloc[:, 2:].values
    return titles, chunks, embeddings

def find_closest_chunks(query_embedding, titles, chunks, embeddings, top_n=5):
    """Find and return top 5 closest chunks ."""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(titles[i], chunks[i], similarities[i]) for i in top_indices]


real_madrid_chunks, real_madrid_embeddings = fetch_and_process_article("Real Madrid")
barcelona_chunks, barcelona_embeddings = fetch_and_process_article("Barcelona")
store_embeddings("football_embeddings.csv",
                 ['Real Madrid']*len(real_madrid_chunks) + ['Barcelona']*len(barcelona_chunks),
                 real_madrid_chunks + barcelona_chunks,
                 real_madrid_embeddings + barcelona_embeddings)


st.title("Find Related Football Content")

query = st.text_input("Enter your query about Real Madrid or Barcelona:")
if st.button("Find Similar Paragraphs"):
    if query:
        query_embedding = get_embeddings([query])[0]
        titles, chunks, embeddings = load_embeddings("football_embeddings.csv")
        closest_chunks = find_closest_chunks(query_embedding, titles, chunks, embeddings)
        for title, chunk, score in closest_chunks:
            st.write(f"**{title}** - Score: {score:.4f}")
            st.write(chunk)
    else:
        st.error("Please enter a query to find related content.")

import os
import streamlit as st
import vertexai
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vertexai.preview.language_models import TextEmbeddingModel
from datetime import datetime

# Irepos
training_dir = 'training'
testing_dir = 'testing'
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# vertex variables
project_id = "miracleinternproj1"
location = "us-central1"
model_name = "textembedding-gecko"

#init vertex
vertexai.init(project=project_id, location=location)

model = TextEmbeddingModel.from_pretrained(model_name) 
def get_embeddings(words):
    """embeddings for list"""
    embeddings = model.get_embeddings(words)
    return [embedding.values for embedding in embeddings]

def plot_embeddings(words, embeddings, ax, label, marker, color):
    """plot embeddings"""
    embeddings = np.array(embeddings)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], marker=marker, color=color, label=label, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def main():
    """Streamlit app to get embeddings and save to CSV."""
    st.title("Text Embedding Visualization")

    # plot inital training values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    training_csv_path = os.path.join(training_dir, 'training.csv')
    if os.path.exists(training_csv_path):
        df = pd.read_csv(training_csv_path)
        all_words = df.values.flatten()  
        all_words = [word for word in all_words if pd.notna(word)]  #Nan
        embeddings = get_embeddings(all_words)

        # Save embeddings
        training_vectors_path = os.path.join(training_dir, 'training_embeddings.csv')
        with open(training_vectors_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for word, embedding in zip(all_words, embeddings):
                writer.writerow([word] + list(embedding))

        plot_embeddings(all_words, embeddings, ax, 'Training Data', 'o', 'blue')
        st.write(f"Embeddings for training data saved to {training_vectors_path}.")
    else:
        st.error("Training data file not found.")

    # test data
    user_text = st.text_input("Enter comma-separated words for testing:")
    download_button = st.button("Get Embedding and Download CSV for Testing")
    if download_button and user_text:
        new_words = [word.strip() for word in user_text.split(",")]
        new_embeddings = get_embeddings(new_words)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_embeddings_{timestamp}.csv"
        testing_csv_path = os.path.join(testing_dir, filename)
        with open(testing_csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for word, embedding in zip(new_words, new_embeddings):
                writer.writerow([word] + list(embedding))

        plot_embeddings(new_words, new_embeddings, ax, 'Testing Data', '^', 'red')
        st.pyplot(fig)
        
        st.success(f"Test embeddings saved to {testing_csv_path}.")
        st.download_button("Download Test CSV", data=open(testing_csv_path, 'rb'), file_name=filename)

if __name__ == "__main__":
    main()

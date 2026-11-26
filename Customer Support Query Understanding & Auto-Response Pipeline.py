
import streamlit as st
import tensorflow as tf
import pickle
import os
import pandas as pd
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. PAGE SETTINGS

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Customer Support Chatbot")
st.write("Welcome! Ask me anything about customer support.")


# 2. FILE PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

DATA_PATH = os.path.join(
    BASE_DIR,
    "Bitext_Customer_Support_Dataset.csv"
)

MAX_LEN = 50


# 3. TEXT CLEANING

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# 4. NORMALIZE CATEGORY AND INTENT

def normalize_label(value):

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value
    )

    return value.strip("_")


# 5. LOAD MODEL AND FILES

@st.cache_resource
def load_project():

    # Load trained model

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Load tokenizer

    with open(TOKENIZER_PATH, "rb") as f:

        tokenizer = pickle.load(f)

    # Load label encoder

    with open(ENCODER_PATH, "rb") as f:

        label_encoder = pickle.load(f)

    # Load dataset

    data = pd.read_csv(DATA_PATH)

    # Normalize column names

    data.columns = (
        data.columns.str.strip().str.lower()
    )

    # Required columns

    required_columns = [
        "instruction",
        "category",
        "intent",
        "response"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    # Remove missing values

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # Clean instructions

    data["instruction_clean"] = (
        data["instruction"].apply(clean_text)
    )

    # Create normalized category column

    data["category_key"] = (
        data["category"].apply(normalize_label)
    )

    # Create normalized intent column

    data["intent_key"] = (
        data["intent"].apply(normalize_label)
    )

    # Remove empty instructions

    data = data[
        data["instruction_clean"] != ""
    ].reset_index(drop=True)

    # Create TF-IDF vectors

    vectorizer = TfidfVectorizer()

    instruction_vectors = vectorizer.fit_transform(
        data["instruction_clean"]
    )

    return (
        model,
        tokenizer,
        label_encoder,
        data,
        vectorizer,
        instruction_vectors
    )


# 6. LOAD PROJECT

try:

    (
        model,
        tokenizer,
        label_encoder,
        data,
        vectorizer,
        instruction_vectors
    ) = load_project()

    st.success("Chatbot loaded successfully!")

except Exception as e:

    st.error(f"Project loading error: {e}")

    st.stop()


# 7. PREDICT INTENT

def predict_intent(message):

    # Clean user question

    cleaned_message = clean_text(message)

    # Convert text into sequences

    sequence = tokenizer.texts_to_sequences(
        [cleaned_message]
    )

    # Apply padding

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict intent

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Get predicted class

    intent_id = prediction.argmax(axis=1)[0]

    # Get confidence score

    confidence = float(
        prediction[0][intent_id]
    )

    # Convert numerical label into intent

    predicted_intent = label_encoder.inverse_transform(
        [intent_id]
    )[0]

    return str(predicted_intent).strip(), confidence


# 8. FIND RESPONSE FROM DATASET

def get_response(message):

    # Predict intent

    predicted_intent, confidence = predict_intent(
        message
    )

    # Normalize predicted intent

    intent_key = normalize_label(
        predicted_intent
    )

    # Match predicted intent with dataset

    matching_rows = data[
        data["intent_key"] == intent_key
    ].copy()

    # Check whether intent exists

    if matching_rows.empty:

        return (
            f"**Predicted Intent:** {predicted_intent}\n\n"
            f"**Confidence Score:** {confidence:.2%}\n\n"
            "Sorry, I couldn't find a matching response. "
            "Please try asking in a different way."
        )

    # Convert user question into TF-IDF vector

    question_vector = vectorizer.transform(
        [clean_text(message)]
    )

    # Get matching dataset indexes

    matching_indices = matching_rows.index.tolist()

    # Select vectors belonging to predicted intent

    matching_vectors = instruction_vectors[
        matching_indices
    ]

    # Calculate cosine similarity

    similarities = cosine_similarity(
        question_vector,
        matching_vectors
    ).flatten()

    # Find the most similar question

    best_position = similarities.argmax()

    best_index = matching_indices[
        best_position
    ]

    # Get similarity score

    best_similarity = float(
        similarities[best_position]
    )

    # Retrieve original dataset response

    answer = data.loc[
        best_index,
        "response"
    ]

    # Retrieve category

    category = data.loc[
        best_index,
        "category"
    ]

    # Retrieve matched instruction

    matched_question = data.loc[
        best_index,
        "instruction"
    ]

    # Final chatbot response

    final_response = (
        f"**Predicted Category:** {category}\n\n"

        f"**Predicted Intent:** {predicted_intent}\n\n"

        f"**Confidence Score:** {confidence:.2%}\n\n"

        f"**Question Similarity:** {best_similarity:.2%}\n\n"

        f"**Matched Dataset Question:** {matched_question}\n\n"

        f"**Answer:** {answer}"
    )

    return final_response


# 9. CHAT HISTORY

if "messages" not in st.session_state:

    st.session_state.messages = []


# 10. DISPLAY PREVIOUS MESSAGES

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# 11. USER INPUT

user_message = st.chat_input(
    "Type your question here..."
)

if user_message:

    # Store user message

    st.session_state.messages.append({

        "role": "user",

        "content": user_message

    })

    # Generate chatbot response

    with st.spinner("Thinking..."):

        try:

            response = get_response(
                user_message
            )

        except Exception as e:

            response = (
                f"Sorry, something went wrong: {e}"
            )

    # Store assistant response

    st.session_state.messages.append({

        "role": "assistant",

        "content": response

    })

    st.rerun()


# 12. RESET CHAT

if st.button(
    "🔄 Start New Conversation",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


# 13. DATASET INFORMATION

with st.expander("Dataset Information"):

    st.write(
        "Total Dataset Records:",
        len(data)
    )

    st.write(
        "Total Categories:",
        data["category"].nunique()
    )

    st.write(
        "Total Intents:",
        data["intent"].nunique()
    )

    st.write(
        "Available Categories:"
    )

    st.write(
        sorted(data["category"].unique())
    )

    st.write(
        "Available Intents:"
    )

    st.write(
        sorted(data["intent"].unique())
    )

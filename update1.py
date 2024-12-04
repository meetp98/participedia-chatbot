import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the cleaned datasets
cases_df = pd.read_csv("cleaned_cases.csv")
methods_df = pd.read_csv("cleaned_methods.csv")
organizations_df = pd.read_csv("cleaned_organizations.csv")

# Combine text for vectorization
all_text = pd.concat([
    cases_df['title'] + " " + cases_df['description'],
    methods_df['title'] + " " + methods_df['description'],
    organizations_df['title'] + " " + organizations_df['description']
], ignore_index=True).fillna("")

# Generate TF-IDF vectors
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(all_text)

# Define chatbot logic
def chatbot_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, text_vectors)
    max_score_idx = similarity_scores.argmax()

    if similarity_scores[0, max_score_idx] < 0.2:
        return "I'm sorry, I can only provide information related to participatory democracy cases, methods, and organizations."

    if max_score_idx < len(cases_df):
        result = cases_df.iloc[max_score_idx]
        return f"Case: {result['title']} - {result['description']} [Link: {result['url']}]"
    elif max_score_idx < len(cases_df) + len(methods_df):
        idx = max_score_idx - len(cases_df)
        result = methods_df.iloc[idx]
        return f"Method: {result['title']} - {result['description']} [Link: {result['url']}]"
    else:
        idx = max_score_idx - len(cases_df) - len(methods_df)
        result = organizations_df.iloc[idx]
        return f"Organization: {result['title']} - {result['description']} [Link: {result['url']}]"

# Streamlit UI
st.set_page_config(page_title="Participatory Democracy Chatbot", layout="wide")
st.title("Participatory Democracy Chatbot Dashboard 🤖")
st.write("Welcome! I can help you explore participatory democracy cases, methods, and organizations.")

# Greet user and provide options
st.sidebar.header("Choose a Category")
category = st.sidebar.selectbox(
    "Select a category to get started:",
    ["General", "Cases", "Methods", "Organizations"]
)

# Provide instructions based on category
if category == "General":
    st.subheader("Welcome to the Participatory Democracy Chatbot!")
    st.write("Ask me anything about participatory democracy. For example:")
    st.write("- What is participatory budgeting?")
    st.write("- How do citizens deliberate?")
elif category == "Cases":
    st.subheader("Ask me about Cases!")
    st.write("Examples of questions:")
    st.write("- Tell me about participatory budgeting.")
    st.write("- What is the case about community engagement?")
elif category == "Methods":
    st.subheader("Ask me about Methods!")
    st.write("Examples of questions:")
    st.write("- What is deliberative democracy?")
    st.write("- How does citizen assembly work?")
elif category == "Organizations":
    st.subheader("Ask me about Organizations!")
    st.write("Examples of questions:")
    st.write("- Which organization focuses on youth?")
    st.write("- What organizations promote participatory governance?")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Your question:")

if st.button("Submit"):
    if user_query.strip():
        response = chatbot_response(user_query)
        st.session_state.chat_history.append({"user": user_query, "bot": response})

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")

# Footer
st.sidebar.write("This chatbot is powered by Participedia datasets and Streamlit.")
st.sidebar.write("Explore participatory democracy effortlessly!")

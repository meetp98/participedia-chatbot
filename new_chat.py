
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the cleaned datasets using your specified paths
cases_df = pd.read_csv(r"C:\Users\Meet\Downloads\cleaned_cases.csv")
methods_df = pd.read_csv(r"C:\Users\Meet\Downloads\cleaned_methods.csv")
organizations_df = pd.read_csv(r"C:\Users\Meet\Downloads\cleaned_organizations.csv")
# Test dataset
test_data = [
    {"query": "What is participatory budgeting?", "expected": "Participatory Budgeting"},
    {"query": "Which organization focuses on youth?", "expected": "Youth for Democracy"},
    {"query": "How do citizens deliberate?", "expected": "Deliberative Democracy"},
]

# Accuracy calculation
correct = 0
total = len(test_data)

for test_case in test_data:
    query = test_case["query"]
    expected = test_case["expected"]

    response = chatbot_response(query)
    
    # Check if the expected result is part of the chatbot response
    if expected.lower() in response.lower():
        correct += 1

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")


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

# Streamlit interface
st.title("Participatory Democracy Chatbot")
st.write("Ask me about participatory cases, methods, or organizations!")

# User input
user_query = st.text_input("Your question:")

if user_query:
    response = chatbot_response(user_query)
    st.write("Chatbot:", response)
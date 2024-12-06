import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Participatory Democracy Chatbot", layout="wide")

# Load the cleaned datasets
cases_df = pd.read_csv("cleaned_cases.csv")
methods_df = pd.read_csv("cleaned_methods.csv")
organizations_df = pd.read_csv("cleaned_organizations.csv")

# Combine datasets for embedding and TF-IDF
cases_texts = (cases_df['title'] + " " + cases_df['description']).fillna("").tolist()
methods_texts = (methods_df['title'] + " " + methods_df['description']).fillna("").tolist()
organizations_texts = (organizations_df['title'] + " " + organizations_df['description']).fillna("").tolist()
all_texts = cases_texts + methods_texts + organizations_texts

# Precompute SentenceTransformer embeddings and cache them
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_texts, convert_to_tensor=True)
    return model, embeddings

model, all_embeddings = load_model_and_embeddings()

# Generate TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(all_texts)

# Categorize the dataset for referencing
case_count = len(cases_texts)
method_count = len(methods_texts)

# Combined Search Function: Semantic + TF-IDF with Suggestions
def combined_search_with_suggestions(query, semantic_weight=0.7, tfidf_weight=0.3, top_k=1, suggestion_k=3):
    # Encode the query using SentenceTransformer
    query_embedding = model.encode(query, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_embedding, all_embeddings)[0]  # Semantic similarity scores

    # Compute TF-IDF cosine similarity
    query_vector = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_vectors)[0]  # TF-IDF cosine similarity scores

    # Combine the scores
    combined_scores = semantic_weight * semantic_scores + tfidf_weight * tfidf_scores
    top_results = combined_scores.topk(k=top_k)  # Get top-k main results

    # Primary result
    main_result = None
    if top_results.values[0].item() > 0.5:  # Strict threshold for the main result
        idx = top_results.indices[0].item()
        score = top_results.values[0].item()

        if idx < case_count:
            result = cases_df.iloc[idx]
            main_result = f"Case: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})", "cases"
        elif idx < case_count + method_count:
            result = methods_df.iloc[idx - case_count]
            main_result = f"Method: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})", "methods"
        else:
            result = organizations_df.iloc[idx - case_count - method_count]
            main_result = f"Organization: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})", "organizations"

    # Suggestions
    suggestions = []
    if main_result:
        _, category = main_result
        top_suggestions = combined_scores.topk(k=suggestion_k + 1)  # Get top-k suggestions (excluding main result)

        for score, idx in zip(top_suggestions.values, top_suggestions.indices):
            idx = idx.item()
            score = score.item()
            if score < 0.5:  # Ensure relevance
                continue

            if category == "cases" and idx < case_count:
                result = cases_df.iloc[idx]
                suggestions.append(f"Case: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})")
            elif category == "methods" and case_count <= idx < case_count + method_count:
                result = methods_df.iloc[idx - case_count]
                suggestions.append(f"Method: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})")
            elif category == "organizations" and idx >= case_count + method_count:
                result = organizations_df.iloc[idx - case_count - method_count]
                suggestions.append(f"Organization: {result['title']} - {result['description']} [Link: {result['url']}] (Score: {score:.2f})")

    return main_result, suggestions

# Streamlit UI

st.title("Participatory Democracy Chatbot 🤖")
st.write("Welcome! This chatbot uses advanced semantic and word-based matching to provide accurate answers.")

# User input for semantic search
st.subheader("Ask Your Question")
user_query = st.text_input("Your question:")

if st.button("Submit"):
    if user_query.strip():
        main_result, suggestions = combined_search_with_suggestions(user_query)

        # Display main result
        if main_result:
            st.write("**Answer:**")
            st.markdown(main_result[0])

            # Display related suggestions
            if suggestions:
                st.write("**Related Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
        else:
            st.markdown("I'm sorry, I couldn't find a relevant answer.")

# Footer
st.sidebar.write("This chatbot is powered by Participedia datasets and Streamlit.")
st.sidebar.write("Explore participatory democracy effortlessly!")

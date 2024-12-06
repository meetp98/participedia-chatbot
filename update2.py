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

# Prebuilt questions categorized
categories = {
    "General": {
        "What is Participedia and how does it work?":
            "Participedia is a platform for sharing and collecting information on inclusive civic engagement and democratic innovations around the world. More info: https://participedia.net/about",
        "How can I contribute to Participedia?":
            "You can contribute by adding new case studies, methods, or organizations. More info: https://participedia.net/getting-started",
        "Can I use Participedia for research or teaching?":
            "Yes, Participedia is a valuable resource for researchers and educators. Explore: https://participedia.net/teaching"
    },
    "Cases": {
        "What is participatory budgeting, and how is it implemented?":
            "Participatory budgeting is a democratic process in which community members directly decide how to allocate part of a public budget. Learn more: https://participedia.net/case/5524",
        "Can you tell me about a case where citizen engagement improved governance?":
            "One example is participatory budgeting in Porto Alegre, Brazil, which improved resource allocation. More info: https://participedia.net/case/5524",
        "What is an example of participatory democracy in education?":
            "One example is the 'Student Voice Committee' in New Zealand. Learn more: https://participedia.net/case/4196"
    },
    "Methods": {
        "What is deliberative democracy?":
            "Deliberative democracy involves public decisions made through thoughtful discussions. Learn more: https://participedia.net/case/485",
        "How does a citizen assembly work?":
            "A citizen assembly brings together randomly selected citizens to discuss and make decisions. More info: https://participedia.net/case/5166",
        "What are participatory budgeting methods?":
            "Participatory budgeting methods include forums, voting, and direct engagement. Example: https://participedia.net/case/44"
    },
    "Organizations": {
        "What organizations promote participatory governance globally?":
            "Organizations like Participatory Budgeting Project and IAP2 promote participatory governance. More info: https://participedia.net/organization/4377",
        "Can you tell me about organizations working on environmental democracy?":
            "Examples include the Environmental Democracy Index and the International Union for Conservation of Nature. More info: https://participedia.net/organization/1053"
    }
}

# Combined Search Function: Semantic + TF-IDF with Related Suggestions
def combined_search(query, semantic_weight=0.7, tfidf_weight=0.3, top_k=3, suggestion_threshold=0.5):
    # Encode the query using SentenceTransformer
    query_embedding = model.encode(query, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_embedding, all_embeddings)[0]  # Semantic similarity scores

    # Compute TF-IDF cosine similarity
    query_vector = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_vectors)[0]  # TF-IDF cosine similarity scores

    # Combine the scores
    combined_scores = semantic_weight * semantic_scores + tfidf_weight * tfidf_scores
    top_results = combined_scores.topk(k=top_k)  # Get top-k results

    results = []
    suggestions = []  # Store suggestions for related items

    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        score = score.item()  # Convert tensor to Python float for display

        # Filter results by a minimum threshold
        if score < suggestion_threshold:
            continue

        if idx < case_count:  # Result from cases
            result = cases_df.iloc[idx]
            results.append(f"Case: {result['title']} - {result['description']} [Link: {result['url']}]")
            suggestions.append(f"[Case] {result['title']} - {result['url']}")
        elif idx < case_count + method_count:  # Result from methods
            result = methods_df.iloc[idx - case_count]
            results.append(f"Method: {result['title']} - {result['description']} [Link: {result['url']}]")
            suggestions.append(f"[Method] {result['title']} - {result['url']}")
        else:  # Result from organizations
            result = organizations_df.iloc[idx - case_count - method_count]
            results.append(f"Organization: {result['title']} - {result['description']} [Link: {result['url']}]")
            suggestions.append(f"[Organization] {result['title']} - {result['url']}")

    return results, suggestions

# Streamlit UI

st.title("Participedia Chatbot")
st.write("Welcome! This chatbot uses advanced semantic and word-based matching to provide accurate answers.")

# Dropdown for categories
category = st.selectbox("Choose a category:", ["Choose a category"] + list(categories.keys()))

if category != "Choose a category":
    st.subheader(f"Questions for {category}")
    selected_question = st.selectbox("Choose a question to get started:", ["Choose a question"] + list(categories[category].keys()))

    # User input for custom query
    st.subheader("Ask Your Own Question")
    user_query = st.text_input("Your question:")

    if st.button("Submit"):
        if selected_question != "Choose a question":
            st.write("**Answer:**")
            st.markdown(categories[category][selected_question])
        elif user_query.strip():
            results, suggestions = combined_search(user_query)
            if results:
                st.markdown(f"**Answer:** {results[0]}")  # Show the top result
                if suggestions:
                    st.write("### Related Suggestions:")
                    for suggestion in suggestions:  # Display suggestions with links
                        st.markdown(f"- {suggestion}")
            else:
                st.markdown("I'm sorry, I couldn't find a relevant answer.")

# Footer
st.sidebar.write("This chatbot is powered by Participedia datasets and Streamlit.")
st.sidebar.write("Explore participatory democracy effortlessly!")

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Participatory Democracy Chatbot", layout="wide")

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

# Prebuilt questions categorized
categories = {
    "General": {
        "What is Participedia and how does it work?":
            "Participedia is a platform for sharing and collecting information on inclusive civic engagement and democratic innovations around the world. Users can browse case studies, methods, and organizations related to participatory governance. More information can be found here: https://participedia.net/about",
        "How can I contribute to Participedia?":
            "You can contribute to Participedia by adding new case studies, methods, or organizations, or by editing and improving existing content. Find out more about how to contribute here: https://participedia.net/getting-started",
        "Can I use Participedia for my research or teaching?":
            "Yes, Participedia is a valuable resource for researchers and educators looking for examples of democratic innovations. The platform offers a repository of information on various participatory processes that can be accessed and used for academic purposes. Explore the research and teaching tools available on Participedia here: https://participedia.net/teaching"
    },
    "Cases": {
        "What is participatory budgeting, and how is it implemented?":
            "Participatory budgeting is a democratic process in which community members directly decide how to allocate part of a public budget. It typically involves a series of meetings and deliberations where residents propose and vote on projects to be funded with public money. Learn more: https://participedia.net/case/5524",
        "Can you tell me about a case where citizen engagement improved governance?":
            "One example is the participatory budgeting process in Porto Alegre, Brazil. This initiative allowed citizens to directly participate in deciding how municipal funds were allocated, leading to better allocation of resources and increased trust in government. Learn more: https://participedia.net/case/5524",
        "What is an example of participatory democracy in education?":
            "One example is the 'Student Voice Committee' in New Zealand, where students collaborate with staff to provide feedback for improving the school environment. Learn more: https://participedia.net/case/4196"
    },
    "Methods": {
        "What is deliberative democracy?":
            "Deliberative democracy involves public decisions made through thoughtful discussions among citizens. For example, the G1000 in Belgium brought together randomly selected citizens to deliberate on policy issues. Learn more: https://participedia.net/case/485",
        "How does a citizen assembly work?":
            "A citizen assembly is a deliberative process that brings together randomly selected citizens to discuss and make decisions on a particular issue. For example: https://participedia.net/case/5166",
        "What are participatory budgeting methods?":
            "Participatory budgeting methods include deliberative forums, voting, and citizen engagement. An example is the 'Porto Alegre Participatory Budgeting Process' in Brazil: https://participedia.net/case/44"
    },
    "Organizations": {
        "What organizations promote participatory governance globally?":
            "Organizations like the Participatory Budgeting Project (https://participedia.net/organization/4377) and IAP2 (https://participedia.net/organization/231) promote participatory governance globally.",
        "Can you tell me about organizations working on environmental democracy?":
            "Examples include the Environmental Democracy Index (https://participedia.net/en/organizations/international-union-conservation-nature-iucn) and the International Union for Conservation of Nature (https://participedia.net/organization/1053)."
    }
}

# Define chatbot logic for user input
def chatbot_response(user_query):
    # TF-IDF logic for custom questions
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

st.title("Participatory Democracy Chatbot ")
st.write("Welcome! Select a category and ask a question or type your own.")

# Dropdown for categories
category = st.selectbox("Choose a category:", ["Choose a category"] + list(categories.keys()))

# Display prebuilt questions for the selected category
if category != "Choose a category":
    st.subheader(f"Questions for {category}")
    selected_question = st.selectbox("Choose a question to get started:", ["Choose a question"] + list(categories[category].keys()))

    # User input for custom query
    st.subheader("Ask Your Own Question")
    user_query = st.text_input("Your question:")

    # Submit button
    if st.button("Submit"):
        # Display answer for prebuilt question
        if selected_question and selected_question != "Choose a question":
            st.write("**Answer:**")
            st.markdown(categories[category][selected_question])
        # Display answer for custom query
        elif user_query.strip():
            st.write("**Answer:**")
            st.markdown(chatbot_response(user_query))

# Footer
st.sidebar.write("This chatbot is powered by Participedia datasets and Streamlit.")
st.sidebar.write("Explore participatory democracy effortlessly!")

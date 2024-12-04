Participatory Democracy Chatbot
Overview
The Participatory Democracy Chatbot is an AI-powered application designed to provide users with precise, summarized, and reliable information about participatory democracy. It assists users in querying data related to Cases, Methods, and Organizations, leveraging structured datasets and advanced natural language processing techniques.

Features
Category Selection: Users can choose between Cases, Methods, or Organizations to narrow their queries.
Accurate Query Resolution: Uses TF-IDF Vectorization and Cosine Similarity to retrieve relevant data.
Summarized Responses: Delivers concise, easy-to-understand summaries of the retrieved information.
Context-Aware: Responds only to queries within the dataset's scope, ensuring accurate and trustworthy answers.
Missing Data Handling: Relies only on available information without imputation, maintaining data credibility.
Suggestions: Provides relevant suggestions when an exact match is unavailable.
Technology Stack
Backend: Python
Framework: Streamlit
Libraries:
pandas for data manipulation
scikit-learn for vectorization and similarity computation
sumy for summarization
Deployment: Can be hosted on platforms like Heroku, AWS, or Google Cloud Platform.

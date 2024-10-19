1. Overview
The Participedia Chatbot provides users with information about participatory democracy by retrieving entries from Participedia’s dataset and generating responses using OpenAI’s GPT models. It implements the RAG (Retrieval-Augmented Generation) technique, combining real data retrieval with AI-generated conversational responses.

2. Features
Accurate Responses: Provides responses based on actual Participedia entries.
RAG Integration: Uses a retrieval component to augment the generative model with relevant data.
Contextual Understanding: Generates responses based on user queries related to participatory democracy.
Validation: Ensures the links and content are valid and related to Participedia’s dataset.

3.Tech Stack
Backend: Node.js, Express.js
AI Models: OpenAI GPT-3.5/GPT-4, OpenAI Embeddings
Database: Vector Database (Pinecone)
API Integration: OpenAI API
Frontend (optional): Basic HTML/CSS/JavaScript interface for user interaction

4.Setup and Installation
Prerequisites
Node.js: v14 or higher
npm: v6 or higher
OpenAI API key
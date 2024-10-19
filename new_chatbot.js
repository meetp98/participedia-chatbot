const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const OpenAI = require('openai');

// Load environment variables
dotenv.config();

// Check if API key exists
if (!process.env.OPENAI_API_KEY) {
  console.error('Error: Missing OpenAI API key. Please check your .env file.');
  process.exit(1);
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const app = express();
app.use(express.json());
app.use(cors());

// Root route to handle the base URL
app.get('/', (req, res) => {
  res.send('Welcome to the Participedia Chatbot!');
});

// Route to handle chatbot queries
app.post('/chat', async (req, res) => {
  const userInput = req.body.message;

  // Example prompt
  const prompt = `Answer the following query using Participedia's dataset only: ${userInput}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 150,
    });

    const reply = response.choices[0].message.content;
    if (reply.includes('Participedia')) {
      res.json({ reply });
    } else {
      res.json({ reply: "Sorry, I can only provide information related to Participedia's dataset." });
    }
  } catch (error) {
    console.error('Error creating chat completion:', error);
    res.status(500).json({ error: 'Failed to get a response from OpenAI.' });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
    
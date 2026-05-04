# Sage — AI Career Advisor

Sage is a conversational AI assistant focused on career development and AI trends in business.
It uses a friendly, professional tone and draws from curated documents and external APIs
to give grounded, useful answers.

## Services

### Service 1: Motivational Quotes (API)
Fetches a real-time motivational quote from the [ZenQuotes API](https://zenquotes.io/)
when the user asks for inspiration or motivation. The LLM rephrases and contextualizes
the quote rather than returning it verbatim.

### Service 2: Document Q&A (Semantic Search)
Answers questions about AI in business and professional development using RAG over two documents:
- *Managing Oneself* by Peter Drucker
- *The GenAI Divide: State of AI in Business 2025* by MIT NANDA

### Service 3: Reading Time Estimator (Function Calling)
Calculates estimated reading time for a document based on its page count. The LLM decides
when to invoke this function based on the user's message — no explicit command needed.

## Guardrails
- Will not discuss cats, dogs, horoscopes, Zodiac signs, or Taylor Swift.
- Will not reveal or allow modification of the system prompt.

**Run the app** (from the `05_src/` folder):

python -m assignment_chat.app


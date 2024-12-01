

## AI Chatbot Application

### Overview

This is an AI-powered chatbot built using Streamlit for the frontend and Python for backend operations. The chatbot answers questions based on the Student Resource Book of the NMIMS Global Access School For Continuing Education, using Google’s Gemini API for natural language generation. It preprocesses a PDF document to extract relevant information, which is then used to provide accurate responses to user queries.

### Features

	•	Natural Language Understanding: Processes user queries and generates responses.
	•	PDF Integration: Extracts and preprocesses data from a PDF document for context-aware answers.
	•	Efficient Search: Uses FAISS for fast and accurate retrieval of relevant information.
	•	Customizable Prompting: Fine-tuned prompts ensure accurate, context-aware answers.
	•	Streamlit UI: Interactive interface for seamless user experience.

### Prerequisites

Before running the application, ensure the following tools and dependencies are installed:
	•	Python 3.8 or higher
	•	Node.js (for running Streamlit’s optional frontend enhancements)
	•	Google Cloud API Key with access to the Generative Language API.

### Setup Instructions

1. Clone the Repository

git clone <repository-url>
cd <repository-folder>

2. Install Backend Dependencies

Install the required Python libraries:

pip install -r requirements.txt

3. Configure Environment Variables

Create a .env file in the root directory to store your Google API Key:

GEMINI_API_KEY=your_google_api_key

4. Place Your PDF File

Add your input.pdf file to the project root or update the PDF_PATH in the code with the correct file path.

5. Run the Streamlit App

Start the Streamlit application:

streamlit run app.py

### Usage

	1.	Enter a Query:
	•	Use the text box in the app to input a question about the content in the PDF.
	2.	Chat History:
	•	View responses and previous conversations in the chat window.
	3.	Real-Time Processing:
	•	The app processes the query, retrieves relevant information, and generates an answer.

### File Structure

├── app.py                     # Streamlit app code
├── main.py                    # Backend processing logic
├── requirements.txt           # Python dependencies
├── input.pdf                  # PDF file for processing
├── chunks.json                # Preprocessed text chunks
├── embeddings.npy             # Precomputed embeddings
├── .env                       # Environment variables (e.g., API key)
├── README.md                  # Documentation



### How It Works

1. Preprocessing

	•	Extracts text from the PDF using PyPDF2.
	•	Splits the text into manageable chunks and generates embeddings using SentenceTransformers.
	•	Stores chunks and embeddings for efficient retrieval.

2. Query Handling

	•	User query is embedded and matched against preprocessed data using FAISS.
	•	Relevant chunks are passed to the Gemini API for response generation.

3. Response Generation

	•	The system generates responses based on the context provided by the relevant chunks.

### Environment Variables

	•	GEMINI_API_KEY: The API key for accessing Google Generative Language API.


Containerization

For production, use Docker:

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

Limitations

	•	Requires an active internet connection to use the Google API.
	•	Responses depend on the accuracy and completeness of the PDF content.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Developed by [Your Name]. Contributions and feedback are welcome!

Replace placeholders (e.g., <repository-url> and your_google_api_key) with actual values for your project. Let me know if you need further assistance!
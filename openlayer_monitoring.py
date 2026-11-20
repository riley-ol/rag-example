"""This is script that simulates running your system in production, to stream
your system's data and metadata to Openlayer.

You can fill in the values for the environment variables in the `.env` file and then
run this script.

The model is imported from the `app.model.pdf_chatbot` module, which is
where we defined our model (the `PDFChatbotModel` class, which uses RAG with LangChain).
Since the Openlayer callback handler is passed to the model and all relevant functions
are decorated with `@trace()`, the model will be traced and data will be sent to
Openlayer.
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)  # .env (riley-demo) wins over shell vars

# Check for required environment variables and prompt if missing
required_env_vars = {
    "OPENLAYER_API_KEY": "Enter your Openlayer API key: ",
    "OPENAI_API_KEY": "Enter your OpenAI API key: ",
    "OPENLAYER_INFERENCE_PIPELINE_ID": "Enter your Openlayer inference pipeline ID: ",
}

for var, prompt in required_env_vars.items():
    if not os.getenv(var):
        value = input(prompt)
        os.environ[var] = value

from app.model import pdf_chatbot

if __name__ == "__main__":

    model = pdf_chatbot.PDFChatbotModel()
    
    # Test the chatbot with a sample question about a sample PDF
    output = model.chat_with_pdf("./data/cv_1.pdf", "What is this document about?")

    print("Question: What is this document about?")
    print(f"Answer: {output.answer}")
    if output.sources:
        print(f"Sources: {len(output.sources)} relevant chunks found")

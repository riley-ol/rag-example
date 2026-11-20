"""Module for our PDF RAG chatbot, which answers questions about uploaded PDFs.

To implement RAG, we use LangChain with vector embeddings and FAISS for retrieval.
We also use Openlayer's Python SDK `trace` and the `OpenlayerHandler` to trace all 
the calls to the model.
"""

import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from openlayer.lib import trace
from openlayer.lib.integrations.langchain_callback import OpenlayerHandler


# This defines the Pydantic model we want our model to return
class ChatResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question based on the PDF content")
    sources: Optional[List[str]] = Field(
        description="List of relevant text snippets from the PDF that were used to answer the question",
        default=None
    )


class PDFChatbotModel:
    """The PDF RAG chatbot model.

    This class wraps around a LangChain RAG chain and answers questions about
    uploaded PDFs using retrieval-augmented generation.
    """

    def __init__(self, model_provider: str = "openai"):
        """Make sure you import / instantiate your system here."""
        # Instantiate the Openlayer callback handler and pass it to the model for tracing
        self.openlayer_handler = OpenlayerHandler()
        
        if model_provider == "openai":
            self.llm = ChatOpenAI(
                # LangChain's default (gpt-3.5-turbo) is retired; pin a current, cheap model.
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                callbacks=[self.openlayer_handler],
                temperature=0
            )
            self.embeddings = OpenAIEmbeddings()
        elif model_provider == "azure":
            self.llm = AzureChatOpenAI(
                callbacks=[self.openlayer_handler],
                temperature=0
            )
            self.embeddings = OpenAIEmbeddings()
        else:
            raise ValueError(f"Invalid model provider: {model_provider}")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Vector store will be created when PDF is loaded
        self.vectorstore = None
        self.retrieval_chain = None

    # Decorate the functions you'd like to trace with @trace()
    @trace()
    def load_pdf(self, file_dir: str) -> List[str]:
        """Loads the PDF, splits it into chunks, and creates a vector store."""
        # Load PDF
        loader = PyPDFLoader(file_dir)
        documents = loader.load()
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create the vector store once and reuse it: embedding every upload is our biggest cost
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        # Create retrieval chain
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context provided, just say that you don't know, 
        don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Please provide a helpful and accurate answer based on the information in the PDF:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            callbacks=[self.openlayer_handler]
        )
        
        return [doc.page_content for doc in texts]

    @trace()
    def ask_question(self, question: str) -> ChatResponse:
        """Ask a question about the loaded PDF."""
        if self.retrieval_chain is None:
            raise ValueError("No PDF loaded. Please load a PDF first using load_pdf()")
        
        # Get response from retrieval chain
        result = self.retrieval_chain.invoke({"query": question})
        
        # Extract answer and sources
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        sources = [doc.page_content[:200] + "..." if len(doc.page_content) > 200 
                  else doc.page_content for doc in source_docs]
        
        return ChatResponse(answer=answer, sources=sources if sources else None)

    @trace()
    def chat_with_pdf(self, file_dir: str, question: str) -> ChatResponse:
        """Load a PDF and answer a question about it (convenience method)."""
        self.load_pdf(file_dir)
        return self.ask_question(question)

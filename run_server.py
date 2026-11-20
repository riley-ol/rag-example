#!/usr/bin/env python3
"""
Simple launcher script to run the FastAPI server from the root directory.
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting PDF RAG Chatbot server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
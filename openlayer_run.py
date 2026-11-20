"""This file is used by Openlayer to run the model and get its outputs.

If you go to openlayer.json, you'll see that the `batchCommand` field points here.

The openlayer_run.py must contain 2 things:

1. A class that inherits from OpenlayerModel and implements the `run` method.
2. A `__main__` block that creates an instance of the class and calls the `run_from_cli`
method.

The model is imported from the `.app.model.pdf_chatbot` module, which is
where we defined our model (the `PDFChatbotModel` class, which uses RAG with
OpenAI embeddings and FAISS vector store).
"""

from openlayer.lib.core.base_model import OpenlayerModel, RunReturn

from app.model import pdf_chatbot


class PDFChatbotModelInterface(OpenlayerModel):
    """Inherits from OpenlayerModel and implements the `run` method."""

    def __init__(self):
        """Make sure you import / instantiate your system here."""
        self.model = pdf_chatbot.PDFChatbotModel()

    def run(
        self,
        file_dir: str,
        question: str,
    ) -> RunReturn:
        """Method that runs the model on a single row of the dataset
        and returns the result (a `RunReturn` object)."""
        response: pdf_chatbot.ChatResponse = self.model.chat_with_pdf(
            file_dir=file_dir, question=question
        )
        output = response.model_dump_json()
        return RunReturn(output=output, other_fields={})


if __name__ == "__main__":
    model = PDFChatbotModelInterface()
    model.run_from_cli()

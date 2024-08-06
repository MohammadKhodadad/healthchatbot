import os
import openai
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from datasets import load_dataset
from langchain.docstore.document import Document

def load_data():
    dataset = load_dataset('omi-health/medical-dialogue-to-soap-summary')['train']
    scenarios=dataset['soap']
    return scenarios

class RAG:
    def __init__(self, faiss_index_path='faiss_index'):

        # Convert string files into documents
        self.conversation_history = []
        documents=load_data()
        self.document_list = [{"text": doc, "metadata": {"source": f"doc_{i}"}} for i, doc in enumerate(documents)]

        # Initialize text splitter and split documents
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.split_docs = []
        for doc in self.document_list:
            splits = text_splitter.split_text(doc['text'])
            for split in splits:
                self.split_docs.append({"text": split, "metadata": doc['metadata']})
        self.documents = [Document(page_content=doc['text'], metadata=doc['metadata']) for doc in self.split_docs]
        print(f'documents are loaded. We have {len(self.documents)} documents now. Let us embed.')

        self.faiss_index_path = faiss_index_path

        # Initialize embeddings and vectorstore
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if os.path.exists(self.faiss_index_path):
            print('embedding loaded from disk')
            self.vectorstore = FAISS.load_local(self.faiss_index_path,self.embeddings,allow_dangerous_deserialization=True)
        else:
            print('embedding getting downloaded.')
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.vectorstore.save_local(self.faiss_index_path)
            print('embedding saved on disk')

        
        print('embedding finished.')

        self.retrieval_chain = RetrievalQA.from_llm(
            retriever=self.vectorstore.as_retriever(),
            llm=OpenAI(temperature=0.7),
        )
        self.instruction_for_feedback="You are a medical advisor, based on the query, History of pieces of information for the person and the documents,  what should be asked from the person. Also you can have some recommendations. but further questions that could help gather critical information is the priority. if there is no query, output empty text like '' ."
        self.instruction_for_extraction="You are a medical advisor, based on the query and the documents, extract the piece of information existing in the input that is related to the persons health. if there is nothing output empty"
    def ask_question(self, question):
        if len(question)>4:
            prompt = f"{self.instruction_for_feedback}\n\nQuestion: {question}\nHistory of pieces of information: {self.conversation_history}\nAnswer:"
            result = self.retrieval_chain(prompt)
            prompt = f"{self.instruction_for_extraction}\n\nQuestion: {question}\nAnswer:"
            extracted_info = self.retrieval_chain(prompt)
            self.add_to_history(extracted_info['result'])
            return result['result']
        else:
            return ''

    def add_to_history(self, result):
        self.conversation_history.append({
            'info': result,
        })
        
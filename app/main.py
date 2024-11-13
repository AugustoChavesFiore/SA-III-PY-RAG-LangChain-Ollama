from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import shutil
import os

app = FastAPI()


class Question(BaseModel):
    question: str

# Define el modelo y parámetros globales para embeddings
llm = ChatOllama(model="llama3.2:1b")
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

# Define el template de respuesta en español
custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
Si la respuesta no se encuentra en dicha información, di que no sabes la respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelve la respuesta útil a continuación y nada más. Responde siempre en español
Respuesta útil:
"""
prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

qa = None
def stream_response(response):
    for line in response:
        yield line


# Endpoint para subir PDF
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")

    # Guardar el archivo PDF temporalmente
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Procesar el PDF
    loader = PyMuPDFLoader(file_path)
    data_pdf = loader.load()
    # Definimos el tamaño de los chunks y el overlap (superposición de los chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

    # Dividimos el contenido del pdf en chunks
    chunks = text_splitter.split_documents(data_pdf)

    # Definir el directorio donde se va a guardar la base de datos
    persist_db = "chroma_db_dir"
    collection_db = "chroma_collection"

    # Crear la base de datos con los chunks
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=persist_db,
        collection_name=collection_db
    )

    # Crear el retriever
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory=persist_db,
        collection_name=collection_db
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 5}  # Cantidad de chunks a retornar
    )
    
    global qa
    # Crear el chain de QA para realizar la búsqueda
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    # Eliminar el archivo PDF temporal
    shutil.rmtree(temp_dir)
    # Retornar mensaje de éxito y el nombre del archivo
    return {"PDF procesado y datos persistidos exitosamente", "nombre: " + file.filename}

# Endpoint para hacer preguntas basadas en el PDF subido
@app.post("/ask_question/")
async def ask_question(question: Question):
    if qa is None:
        raise HTTPException(status_code=400, detail="No se ha cargado ningún PDF")
    response = qa.invoke({"query": question.question})
    return {"answer": response['result']}


@app.post("/ask_question_stream/")
async def ask_question(question: Question):
    if qa is None:
        raise HTTPException(status_code=400, detail="No se ha cargado ningún PDF")
    response = qa.stream({"query": question})
    return StreamingResponse(stream_response(response), media_type="application/json")
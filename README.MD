# Proyecto de FastAPI y Ollama para RAG

Este proyecto es una aplicación web construida con FastAPI que permite a los usuarios cargar archivos PDF y hacer preguntas sobre su contenido utilizando Retrieval-Augmented Generation (RAG) con el modelo `ollama`.

## Requisitos

- Python 3.9 o superior
- Ollama
- Modelo `llama3.2:1b`

## Instalación

1. Clona el repositorio:

```sh
    git clone https://github.com/AugustoChavesFiore/SA-III-PY-RAG-LangChain-Ollama
    cd SA-III-PY-RAG-LangChain-Ollama
```
2. Crea y activa un entorno virtual
```bash
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
```
3. Instala las dependencias
```bash
    pip install -r requirements.txt
```
4. Instala ollama siguiendo las instrucciones de la <a href='https://ollama.com/'> documentación oficial de Ollama</a>.

5. Descarga e instala el modelo llama3.2:1b:
```bash
ollama pull llama3.2:1b
```

6. Correr el proyecto en modo desarrollo
```bash
fastapi dev app/main.py
```

# Uso 
## Subir un PDF
Para subir un archivo PDF, realiza una solicitud POST al endpoint /upload_pdf con el archivo PDF adjunto.
```json
    {
    "file":"...."
    }
```


## Hacer una Pregunta
Para hacer una pregunta sobre el contenido del PDF subido, realiza una solicitud POST al endpoint /ask_question con la pregunta en el cuerpo de la solicitud.

```json
    {
    "question":"Que es c#"
    }
```

## Preguntar si un archivo PDF ha sido subido
Para preguntar si un archivo PDF ha sido subido, realiza una solicitud GET al endpoint /get_pdf_name.

## Archivos Clave
* app/main.py: Contiene la lógica principal de la aplicación FastAPI.
* requirements.txt: Lista de dependencias de Python necesarias para la aplicación.
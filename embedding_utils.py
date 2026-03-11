import os
import hashlib
import re
import io
import time
import tempfile

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

def sanitize_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-=]", "_", key)

# Load environment variables
load_dotenv()

# Initialize Google Generative AI client
client_genai = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
VECTOR_DIMENSIONS = 3072  # gemini-embedding-001 produces 3072-dim vectors


def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_text_from_image(image_bytes):
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        if response.error.message:
            print(f"Google Vision API error: {response.error.message}")
            return ""
        return response.full_text_annotation.text if response.full_text_annotation else ""
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""


def extract_tables_from_page(page):
    tables = []
    tab = page.find_tables()
    if tab.tables:
        for table in tab.tables:
            try:
                df = pd.DataFrame(table.extract())
                table_str = df.to_string(index=False, header=False)
                tables.append(table_str)
            except Exception as e:
                print(f"Table extraction error: {str(e)}")
    return tables


def process_pdf(file_path, file_name):
    document_text = []
    pdf_document = fitz.open(file_path)

    for page_num, page in enumerate(pdf_document):
        page_text = page.get_text()
        tables = extract_tables_from_page(page)
        table_text = "\n\n".join(tables)

        if len(page_text.strip()) < 100:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes()
            ocr_text = extract_text_from_image(img_bytes)
            if ocr_text.strip():
                document_text.append(f"[Page {page_num+1} OCR Text]:\n{ocr_text}")

        if page_text.strip():
            document_text.append(f"[Page {page_num+1} Text]:\n{page_text}")

        if table_text.strip():
            document_text.append(f"[Page {page_num+1} Table]:\n{table_text}")

    pdf_document.close()
    combined_text = "\n\n".join(document_text)
    normalized_text = normalize_text(combined_text)
    return chunk_text(normalized_text)


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return [normalize_text(chunk) for chunk in chunks if chunk.strip()]


def create_embedding(text):
    try:
        result = client_genai.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        embedding = [float(x) for x in result.embeddings[0].values]
        return embedding
    except Exception as e:
        import traceback
        print(f"Error creating embedding: {str(e)}")
        print(traceback.format_exc())
        return [0.0] * VECTOR_DIMENSIONS

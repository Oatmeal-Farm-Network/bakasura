import os
import hashlib
import numpy as np
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from openai import AzureOpenAI
import re


def sanitize_key(key: str) -> str:
    """
    Convert a string into a valid Azure Search document key:
    Only allows letters, digits, underscore (_), dash (-), and equal sign (=)
    """
    return re.sub(r"[^a-zA-Z0-9_\-=]", "_", key)


# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def hash_text(text):
    """Generate a hash for the text to help with deduplication."""
    return hashlib.md5(text.encode()).hexdigest()


def normalize_text(text):
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters and normalize
    text = text.strip()
    return text


def extract_text_from_image(image_bytes):
    """Extract text from an image using Azure Computer Vision OCR."""
    try:
        # Azure Computer Vision OCR implementation
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from azure.cognitiveservices.vision.computervision.models import (
            OperationStatusCodes,
        )
        from msrest.authentication import CognitiveServicesCredentials
        import time

        # Load credentials
        endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        key = os.getenv("AZURE_VISION_KEY")

        # Authenticate client
        vision_client = ComputerVisionClient(
            endpoint, CognitiveServicesCredentials(key)
        )

        # Create a BytesIO object
        image_stream = io.BytesIO(image_bytes)

        # Call the API
        read_response = vision_client.read_in_stream(image_stream, raw=True)

        # Get the operation location (URL with an ID at the end)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Wait for the operation to complete
        while True:
            read_result = vision_client.get_read_result(operation_id)
            if read_result.status not in ["notStarted", "running"]:
                break
            time.sleep(1)

        # Extract the text
        text = ""
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text += line.text + "\n"

        return text
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""


def extract_tables_from_page(page):
    """Extract tables from a PDF page."""
    tables = []

    # Extract tables using PyMuPDF's find_tables method
    tab = page.find_tables()
    if tab.tables:
        for table in tab.tables:
            try:
                # Convert table to DataFrame
                df = pd.DataFrame(table.extract())
                # Convert DataFrame to string
                table_str = df.to_string(index=False, header=False)
                tables.append(table_str)
            except Exception as e:
                print(f"Table extraction error: {str(e)}")

    return tables


def process_pdf(file_path, file_name):
    """
    Process a PDF file to extract text, perform OCR on images,
    and extract tables, then chunk the content.
    """
    document_text = []

    # Open the PDF
    pdf_document = fitz.open(file_path)

    # Process each page
    for page_num, page in enumerate(pdf_document):
        # Extract text from page
        page_text = page.get_text()

        # Extract tables
        tables = extract_tables_from_page(page)
        table_text = "\n\n".join(tables)

        # If page has little text, it might be an image that needs OCR
        if len(page_text.strip()) < 100:
            # Get the page as an image
            pix = page.get_pixmap()
            img_bytes = pix.tobytes()

            # Extract text using OCR
            ocr_text = extract_text_from_image(img_bytes)

            # If OCR found text, add it
            if ocr_text.strip():
                document_text.append(f"[Page {page_num+1} OCR Text]:\n{ocr_text}")

        # Add page text
        if page_text.strip():
            document_text.append(f"[Page {page_num+1} Text]:\n{page_text}")

        # Add table text
        if table_text.strip():
            document_text.append(f"[Page {page_num+1} Table]:\n{table_text}")

    # Close the PDF
    pdf_document.close()

    # Combine all text
    combined_text = "\n\n".join(document_text)

    # Normalize text
    normalized_text = normalize_text(combined_text)

    # Chunk the text
    return chunk_text(normalized_text)


def chunk_text(text):
    """Chunk text with intelligent boundaries."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    return [normalize_text(chunk) for chunk in chunks if chunk.strip()]


def create_embedding(text):
    """Create an embedding for text using Azure OpenAI API."""
    try:
        # Get Azure OpenAI deployment name
        deployment_name = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )

        # Generate embedding using Azure OpenAI
        response = client.embeddings.create(model=deployment_name, input=text)

        # Extract embedding from response
        embedding = response.data[0].embedding

        # Convert to list of floats
        embedding = [float(x) for x in embedding]

        return embedding
    except Exception as e:
        import traceback

        print(f"Error creating embedding: {str(e)}")
        print(traceback.format_exc())
        # Return a zero vector (1536 dimensions for text-embedding-ada-002)
        return [0.0] * 1536

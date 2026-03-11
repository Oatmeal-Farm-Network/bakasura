import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import firestore
from embedding_utils import sanitize_key, VECTOR_DIMENSIONS

load_dotenv()

FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "bakasura-docs")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")


def initialize_search_client():
    """Initialize Firestore client (replaces Azure Search client)."""
    db = firestore.Client(project=GCP_PROJECT_ID, database=os.getenv("FIRESTORE_DATABASE", "charlie"))
    collection_ref = db.collection(FIRESTORE_COLLECTION)
    print(f"✅ Connected to Firestore collection '{FIRESTORE_COLLECTION}'")
    # Return two values to match the original interface (search_client, index_client)
    return collection_ref, db


def store_embedding(collection_ref, text, embedding, metadata, doc_key=None):
    """Store a text chunk and its embedding in Firestore."""
    try:
        text_hash = metadata.get("text_hash")

        # Duplicate check
        if text_hash:
            existing = list(
                collection_ref.where("text_hash", "==", text_hash).limit(1).stream()
            )
            if existing:
                print(f"⚠️ Duplicate content detected. Skipping storage.")
                return False

        if not doc_key:
            doc_key = sanitize_key(
                f"{metadata['filename']}_{metadata['chunk_id']}_{uuid.uuid4().hex[:6]}"
            )

        document = {
            "content": text,
            "content_vector": embedding,
            "filename": metadata.get("filename"),
            "chunk_id": metadata.get("chunk_id"),
            "text_hash": text_hash,
            "timestamp": datetime.fromtimestamp(metadata.get("timestamp", 0)),
            "file_type": "pdf",
            "page_number": metadata.get("page_number"),
            "metadata": json.dumps(metadata),
        }

        collection_ref.document(doc_key).set(document)
        return True

    except Exception as e:
        print(f"❌ Error storing embedding: {e}")
        return False


def get_index_stats(collection_ref, collection_name=None):
    """Get statistics about the Firestore collection."""
    try:
        docs = list(collection_ref.stream())
        total_chunks = len(docs)
        unique_files = len(set(d.to_dict().get("filename") for d in docs))
        return {
            "total_documents": total_chunks,
            "index_name": collection_name or FIRESTORE_COLLECTION,
            "status": "connected",
            "unique_files": unique_files,
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "index_name": collection_name or FIRESTORE_COLLECTION,
            "status": "error",
            "error": str(e),
        }
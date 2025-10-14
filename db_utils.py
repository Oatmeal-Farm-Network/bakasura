# db_utils.py — Migrated from Cosmos DB to Azure AI Search

import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from embedding_utils import sanitize_key

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "bakasura-docs-v3-chunk1000")
VECTOR_DIMENSIONS = 1536


def initialize_search_client():
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        raise ValueError(
            "Azure AI Search endpoint and key must be provided in environment variables"
        )

    credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=credential
    )

    _create_or_update_index(index_client)
    return search_client, index_client


def _create_or_update_index(index_client):
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="content", type=SearchFieldDataType.String, searchable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="my-vector-config",
        ),
        SimpleField(
            name="filename",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="text_hash", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="timestamp",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="file_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="page_number", type=SearchFieldDataType.Int32, filterable=True
        ),
        SearchableField(
            name="metadata", type=SearchFieldDataType.String, searchable=True
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-config", algorithm_configuration_name="my-hnsw-config"
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    index_client.create_or_update_index(index)
    print(f"✅ Azure AI Search index '{SEARCH_INDEX_NAME}' is ready")


def store_embedding(search_client, text, embedding, metadata, doc_key=None):
    try:
        text_hash = metadata.get("text_hash")

        if text_hash:
            existing = list(
                search_client.search(
                    search_text="*",
                    filter=f"text_hash eq '{text_hash}'",
                    select=["id"],
                    top=1,
                )
            )
            if existing:
                print(f"⚠️ Duplicate content detected. Skipping storage.")
                return False

        if not doc_key:
            doc_key = sanitize_key(
                f"{metadata['filename']}_{metadata['chunk_id']}_{uuid.uuid4().hex[:6]}"
            )

        document = {
            "id": doc_key,
            "content": text,
            "content_vector": embedding,
            "filename": metadata.get("filename"),
            "chunk_id": metadata.get("chunk_id"),
            "text_hash": text_hash,
            "timestamp": datetime.fromtimestamp(
                metadata.get("timestamp", 0)
            ).isoformat()
            + "Z",
            "file_type": "pdf",
            "page_number": metadata.get("page_number"),
            "metadata": json.dumps(metadata),
        }

        result = search_client.upload_documents(documents=[document])
        return result[0].succeeded if result else False

    except Exception as e:
        print(f"❌ Error in store_embedding: {e}")
        return False


def get_document_stats(search_client):
    try:
        count_result = search_client.search("*", include_total_count=True, top=0)
        total_chunks = count_result.get_count()

        facet_result = search_client.search("*", facets=["filename"], top=0)
        unique_files = len(facet_result.get_facets().get("filename", []))

        return {"total_chunks": total_chunks, "unique_files": unique_files}

    except Exception as e:
        print(f"⚠️ Error getting stats: {e}")
        return {"total_chunks": 0, "unique_files": 0}

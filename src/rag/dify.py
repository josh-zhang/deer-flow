import logging
import asyncio
import os
from urllib.parse import urlparse
import requests
from src.rag.retriever import Resource, Retriever
from src.config.loader import get_str_env
from src.rag.llm_reranker import get_reranked_chunks

logger = logging.getLogger(__name__)


class DifyProvider(Retriever):
    """
    DifyProvider is a provider that uses dify to retrieve documents.
    """

    api_url: str
    api_key: str
    init_resources: list[Resource]

    def __init__(self):
        api_url = get_str_env("DIFY_API_URL", "http://198.203.120.5/v1")
        if not api_url:
            raise ValueError("DIFY_API_URL is not set")
        self.api_url = api_url

        api_key = os.getenv("DIFY_API_KEY")
        if not api_key:
            raise ValueError("DIFY_API_KEY is not set")
        self.api_key = api_key

        self.init_resources = self.list_resources()

    def query_relevant_documents(
        self, query: str, top_k: int, background: str, resources: list[Resource] = []
    ) -> dict[str, dict]:
        # if not resources:
        #     return []

        if len(query) > 240:
            logger.warning(
                f"query exceed 240 chars, only keeping first 240 chars for querying, original query:\n{query}"
            )
            query = query[:240]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        this_top_k = int((top_k / len(self.init_resources)) + 0.5)

        doc_groups: dict[str, dict] = {}

        for resource in self.init_resources:
            resource_description = resource.description
            dataset_id, _ = parse_uri(resource.uri)

            if not dataset_id.startswith("dify-"):
                continue

            dataset_id = dataset_id[5:]

            payload = {
                "query": query,
                "retrieval_model": {
                    "search_method": "hybrid_search",
                    "reranking_enable": False,
                    "weights": {
                        "weight_type": "customized",
                        "keyword_setting": {"keyword_weight": 0.3},
                        "vector_setting": {"vector_weight": 0.7},
                    },
                    "top_k": this_top_k,
                    "score_threshold_enabled": True,
                    "score_threshold": 0.5,
                },
            }

            response = requests.post(
                f"{self.api_url}/datasets/{dataset_id}/retrieve",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to query documents: {response.text}")

            result = response.json()
            records = result.get("records", {})

            unsorted_list = list()
            for record in records:
                segment = record.get("segment")
                if not segment:
                    unsorted_list.append("空")
                    continue

                document_info = segment.get("document")
                if not document_info:
                    unsorted_list.append("空")
                    continue

                doc_id = document_info.get("id")
                doc_name = document_info.get("name")
                if not doc_id or not doc_name:
                    unsorted_list.append("空")
                    continue

                chunk_content = segment.get("content", "")
                unsorted_list.append(chunk_content)

            ranked_docs = get_reranked_chunks(query, unsorted_list, background)

            ranked_records = list()
            for doc in ranked_docs:
                original_index = doc['original_index']
                score = doc['score']

                if score > 0:
                    # 如果文档 相关 或 不确定
                    record = records[original_index]
                    ranked_records.append(record)

            new_records = ranked_records[:this_top_k]

            for record in new_records:
                segment = record.get("segment")
                chunk_idx = record.get("position")

                if not segment:
                    continue

                document_info = segment.get("document")
                if not document_info:
                    continue

                doc_id = document_info.get("id")
                doc_name = document_info.get("name")
                if not doc_id or not doc_name:
                    continue

                if doc_name not in doc_groups:
                    url = f"{self.api_url}/datasets/{dataset_id}/documents/{doc_id}/segments"
                    doc_groups[doc_name] = {
                        "document_title": doc_name,
                        "document_url": url,
                        "description": resource_description,
                        "chunks": [],
                    }

                doc_groups[doc_name]["chunks"].append({
                    "chunk_index": str(chunk_idx),
                    "chunk_content": segment.get("content", ""),
                })

        return doc_groups

    async def query_relevant_documents_async(
        self, query: str, top_k: int, background: str, resources: list[Resource] = []
    ) -> dict[str, dict]:
        """
        Asynchronous version of query_relevant_documents.
        wraps the synchronous implementation in asyncio.to_thread() to avoid blocking the event loop.
        """
        return await asyncio.to_thread(
            self.query_relevant_documents, query, top_k, background, resources
        )

    def list_resources(self, query: str | None = None) -> list[Resource]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        params = {}
        if query:
            params["keyword"] = query

        response = requests.get(
            f"{self.api_url}/datasets", headers=headers, params=params
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list resources: {response.text}")

        result = response.json()
        resources = []

        for item in result.get("data", []):
            item = Resource(
                uri=f"rag://dataset/dify-{item.get('id')}",
                title=item.get("name", ""),
                description=item.get("description", ""),
            )
            resources.append(item)

        return resources

    async def list_resources_async(self, query: str | None = None) -> list[Resource]:
        """
        Asynchronous version of list_resources.
        wraps the synchronous implementation in asyncio.to_thread() to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self.list_resources, query)


def parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "rag":
        raise ValueError(f"Invalid URI: {uri}")
    return parsed.path.split("/")[1], parsed.fragment

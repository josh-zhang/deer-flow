import os
import logging
from typing import Annotated, Dict, Tuple

import requests

from langchain_core.tools import tool

from src.rag.retriever import Chunk, Document
from src.tools.extractor import extract_relevant_chunks
from src.rag.retriever import (
    format_crawl_fetch_return,
    generate_checksum,
    remove_line_start_hashes,
)

from src.tools.decorators import log_io

logger = logging.getLogger(__name__)


def crawl(url: str, document_category: str) -> Document:
    """Crawl content from the given URL and return a Document object."""
    dify_api_url = os.getenv("DIFY_API_URL")
    dify_api_key = os.getenv("DIFY_API_KEY")

    logger.info(f"URL: {url}")

    if dify_api_url in url:
        headers = {
            "Authorization": f"Bearer {dify_api_key}",
            "Content-Type": "application/json",
        }
        try:
            # Make the HTTP GET request
            response = requests.get(url, headers=headers, timeout=10)  # 10-second timeout for safety
            # Check if the request was successful
            response.raise_for_status()

            result_json = response.json()

            chunk_list = result_json.get("data")
            chunks = []

            document_id = ""
            for chunk in chunk_list:
                idx = str(chunk["position"])
                document_id = chunk["document_id"]
                ct = chunk["content"]
                if not ct:
                    continue
                chunks.append(Chunk(idx, ct))

            if chunks and document_id:
                article = Document(
                    id=document_id,
                    title="",
                    url=url,
                    description=document_category,
                )
                return article
        except Exception as e:
            logger.error(e)
    else:
        try:
            # Make the HTTP GET request with 10-second timeout for safety
            response = requests.get(url, timeout=10)
            # Check if the request was successful
            response.raise_for_status()
            # Get the content as a UTF-8 decoded string
            results = response.json()
            result_content = ""
            title = ""
            chunks = []
            for paragraph in results:
                title = paragraph["title"]
                idx = paragraph["idx"]
                content = paragraph["content"]
                if not content:
                    continue

                content = remove_line_start_hashes(content)
                result_content += content + "\n\n"
                chunks.append(Chunk(idx, content))
            if chunks:
                article = Document(
                    id=generate_checksum(result_content),
                    title=title,
                    url=url,
                    description=document_category,
                    chunks=chunks,
                )
                return article
        except Exception as e:
            logger.error(e)

    return None


@tool(response_format="content_and_artifact")
@log_io
def crawl_tool(
    url: Annotated[str, "The url to crawl"],
    target_questions: Annotated[
        str,
        "Questions to be answered based on the content of the file. Use the question mark (?) as a delimiter to separate individual questions.",
    ],
    document_category: Annotated[str, "The category of the document"] = "",
) -> Tuple[str, Dict]:
    """Use this to crawl a url with target_questions and get relevant content of the target questions in plain text format."""
    error_msg = f"Failed to crawl. 本地文档知识库中没有找到文档: {url}"
    max_content = 8000
    target_questions = target_questions.replace("？", "?")
    target_questions = [i.strip() for i in target_questions.split("?")]
    target_questions = [i.strip() for i in target_questions if i.strip()]
    try:
        document = crawl(url, document_category)
        if document is not None:
            extraction_result = extract_relevant_chunks(
                chunks=document.chunks,
                target_questions=target_questions,
                document_title=document.title,
                context_window=1,
                full_text_threshold=max_content,
            )
            if extraction_result is not None:
                content_md, artifact_dict = format_crawl_fetch_return(
                    extraction_result=extraction_result,
                    tool_type="crawl",
                    document_url=document.url,
                    description=document.description,
                )
        return content_md, artifact_dict
    except BaseException as e:
        error_msg += f"{error_msg}: Error: {repr(e)}"
        logger.error(error_msg)
        return f"提取文档失败 : {error_msg}", {"tool_type": "crawl", "documents": []}

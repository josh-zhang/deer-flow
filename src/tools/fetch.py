import logging
import re

import requests
from urllib.parse import urlencode
from typing import Annotated, Dict, Tuple
from langchain_core.tools import tool

from src.rag.retriever import Chunk, Document, format_crawl_fetch_return, generate_checksum, remove_line_start_hashes
from src.tools.extractor import extract_relevant_chunks
from src.config.loader import get_str_env

from src.tools.decorators import log_io

logger = logging.getLogger(__name__)

current_kb_name = get_str_env("CURRENT_KB_NAME", "80260423_1632377")
mskb_url = get_str_env("MSKB_KNOWLEDGE_BASE_API_URL", "100.203.120.7:40795")


def sanitize_filename(filename: str) -> str:
    r"""Replace invalid characters in the given filename."""
    sanitized = re.sub(r"[\\/*?\"<>|]", "", filename).strip()
    logger.warning(
        f"filename: {filename} got empty string after sanitize_filename, return empty string"
    )
    return sanitized


def is_valid_file_id(s: str) -> bool:
    pattern = r"^\d{6}-\d{6}$"
    return bool(re.fullmatch(pattern, s))


@tool(response_format="content_and_artifact")
@log_io
def fetch_tool(
    file_name: Annotated[
        str, "The file name to fetch, do not include suffix."
    ],
    target_questions: Annotated[
        str,
        "Questions to be answered based on the content of the file. Use the question mark (?) as a delimiter to separate individual questions.",
    ],
) -> Tuple[str, Dict]:
    """Use this to fetch a file with file name (no suffix) and get relevant content of the target questions in plain text format."""

    error_msg = f"Failed to fetch. 本地文档知识库中没有找到文档: {file_name}"

    sanitized_file_title = sanitize_filename(file_name)

    if not sanitized_file_title:
        logger.error(error_msg)
        return (
            "file_name 无效。",
            {"tool_type": "fetch", "documents": []},
        )

    if is_valid_file_id(sanitized_file_title[:13]):
        logger.warning(
            f"sanitized_file_title {sanitized_file_title} is loading with file_id, deleted..."
        )
        sanitized_file_title = sanitized_file_title[13:]

    sanitized_file_title = (
        sanitized_file_title
        if sanitized_file_title.endswith(".html")
        else f"{sanitized_file_title}.html"
    )
    logger.info(
        f"title {file_name} sanitized_filename {sanitized_file_title}")

    parameters = urlencode(
        {"knowledge_base_name": current_kb_name, "file_name": sanitized_file_title}
    )
    url = f"http://{mskb_url}/knowledge_base_list_docs?{parameters}"
    max_content = 8000
    target_questions = target_questions.replace("？", "?")
    target_questions = [q.strip() for q in target_questions.split("?")]
    target_questions = [q.strip() for q in target_questions if q.strip()]

    try:
        # Make the HTTP GET request
        response = requests.get(url, timeout=10)  # 10-second timeout for safety

        # Check if the request was successful
        response.raise_for_status()

        results = response.json()
        result_content = ""
        title = ""
        chunks = []
        for paragraph in results:
            idx = paragraph["idx"]
            title = paragraph["title"]
            content = paragraph["content"]
            if not content:
                continue
            content = remove_line_start_hashes(content)
            result_content += content + "\n"
            chunks.append(Chunk(idx, content))

        if chunks:
            document = Document(
                id=generate_checksum(result_content),
                title=title,
                url=url,
                description="",
                chunks=chunks,
            )
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
                    tool_type="fetch",
                    document_url=document.url,
                    description=document.description,
                )
                return content_md, artifact_dict
    except BaseException as e:
        error_msg = f"{error_msg}: Error: {repr(e)}"
        logger.error(error_msg)
        return (
            f"未找到文档 (名称: {file_name})。",
            {"tool_type": "fetch", "documents": []},
        )

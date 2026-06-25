# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from src.tools.extractor import extract_relevant_chunks
from src.graph.nodes import format_crawl_fetch_return

from .decorators import log_io

logger = logging.getLogger(__name__)


def crawl():
    # TODO
    return []


@tool
@log_io
def crawl_tool(
        url: Annotated[str, "The url to crawl."],
        target_questions: list[str],
):
    try:
        article = crawl(url)
        resutls = extract_relevant_chunks(article.chunks,
                                          target_questions,
                                          article.title,
                                          full_text_threshold=8000)

        content_md, artifect_dict = format_crawl_fetch_return(resutls)

        return content_md, artifect_dict
    except BaseException as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg, {}

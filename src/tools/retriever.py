# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config.tools import SELECTED_RAG_PROVIDER
from src.rag import Document, Resource, Retriever, build_retriever
from src.rag.retriever import format_local_search_return

logger = logging.getLogger(__name__)


class RetrieverInput(BaseModel):
    query: str = Field(description="search keywords to look up")
    background: str = Field(
        default="",
        description="background context of the current search step",
    )


class RetrieverTool(BaseTool):
    name: str = "local_search_tool"
    description: str = "Useful for retrieving information from the file with `rag://` uri prefix, it should be higher priority than the web search or writing code. Input should be a search keywords."
    args_schema: Type[BaseModel] = RetrieverInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    retriever: Retriever = Field(default_factory=Retriever)
    resources: list[Resource] = Field(default_factory=list)
    top_k: int = 10

    def _run(
        self,
        query: str,
        background: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[str, dict]:
        logger.info(
            f"Retriever tool query: {query}", extra={"resources": self.resources}
        )
        documents = self.retriever.query_relevant_documents(
            query, self.top_k, background, self.resources
        )
        if not documents:
            return "No results found from the local knowledge base.", None
        return format_local_search_return(documents)

    async def _arun(
        self,
        query: str,
        background: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> tuple[str, dict]:
        logger.info(
            f"Retriever tool query: {query}", extra={"resources": self.resources}
        )
        documents = await self.retriever.query_relevant_documents_async(
            query, self.top_k, background, self.resources
        )
        if not documents:
            return "No results found from the local knowledge base.", None
        return format_local_search_return(documents)


def get_retriever_tool(
    max_search_results: int,
    report_style: str,
    resources: List[Resource],
) -> RetrieverTool | None:
    if not resources:
        return None
    logger.info(f"create retriever tool: {SELECTED_RAG_PROVIDER}")
    retriever = build_retriever()

    if not retriever:
        return None
    return RetrieverTool(
        retriever=retriever, resources=resources, top_k=max_search_results
    )

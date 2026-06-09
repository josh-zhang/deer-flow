# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from src.config.report_style import ReportStyle
from src.rag.retriever import Resource


class AttachedFile(BaseModel):
    """A file attached to a user message via the /chat input box.

    Text files carry their UTF-8 decoded content in `text`; images carry
    their bytes as base64 in `b64`. Validated at the API boundary; nodes
    can read these directly from state["attached_files"].
    """

    id: str = Field(..., description="Client-generated id (nanoid)")
    message_id: str = Field(
        ..., description="Id of the user message this file was attached to"
    )
    name: str = Field(..., description="Original filename (sanitized server-side)")
    mime: str = Field(..., description="MIME type")
    kind: Literal["text", "image"] = Field(..., description="Attachment kind")
    size_bytes: int = Field(..., ge=0, description="Raw file size in bytes")
    text: Optional[str] = Field(
        None, description="UTF-8 decoded content (required when kind=='text')"
    )
    b64: Optional[str] = Field(
        None, description="Base64-encoded bytes (required when kind=='image')"
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> "AttachedFile":
        if self.kind == "text" and self.text is None:
            raise ValueError("AttachedFile.text is required when kind=='text'")
        if self.kind == "image" and not self.b64:
            raise ValueError("AttachedFile.b64 is required when kind=='image'")
        return self


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    resources: Optional[List[Resource]] = Field(
        [], description="Resources to be used for the research"
    )
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    locale: Optional[str] = Field(
        "en-US", description="Language locale for the conversation (e.g., en-US, zh-CN)"
    )
    max_plan_iterations: Optional[int] = Field(
        1, description="The maximum number of plan iterations"
    )
    max_step_num: Optional[int] = Field(
        3, description="The maximum number of steps in a plan"
    )
    max_search_results: Optional[int] = Field(
        3, description="The maximum number of search results"
    )
    auto_accepted_plan: Optional[bool] = Field(
        False, description="Whether to automatically accept the plan"
    )
    interrupt_feedback: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan"
    )
    mcp_settings: Optional[dict] = Field(
        None, description="MCP settings for the chat request"
    )
    enable_background_investigation: Optional[bool] = Field(
        True, description="Whether to get background investigation before plan"
    )
    enable_web_search: Optional[bool] = Field(
        True, description="Whether to enable web search, set to False to use only local RAG"
    )
    report_style: Optional[ReportStyle] = Field(
        ReportStyle.BANK_BUSINESS_ANALYSIS, description="The style of the report"
    )
    enable_deep_thinking: Optional[bool] = Field(
        False, description="Whether to enable deep thinking"
    )
    enable_clarification: Optional[bool] = Field(
        None,
        description="Whether to enable multi-turn clarification (default: None, uses State default=False)",
    )
    max_clarification_rounds: Optional[int] = Field(
        None,
        description="Maximum number of clarification rounds (default: None, uses State default=3)",
    )
    interrupt_before_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names to interrupt before execution (e.g., ['db_tool', 'api_tool'])",
    )
    attached_files: Optional[List[AttachedFile]] = Field(
        default_factory=list,
        description="Files attached to the current user message (text or image). "
        "Travel with the request and become available to graph nodes via "
        "state['attached_files']. Accumulates across turns of the same thread.",
    )


class CGRequest(BaseModel):
    """CG 合规营销文案生成请求"""
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    product_name: str = Field(..., description="产品名称")
    product_type: str = Field(..., description="产品类型，如联名信用卡、分期产品")
    campaign_name: str = Field(default="", description="营销活动名称（如有）")
    channels: list[str] = Field(default_factory=list, description="渠道列表，如企业微信、全民生活APP、短信")
    personas: list[str] = Field(default_factory=list, description="客群列表，如山姆会员/家庭消费群体")
    scene_empathy: bool = Field(default=True, description="是否启用场景化共情")
    relationship_temperature: str = Field(default="warm", description="关系温度: cold/warm/hot/rm_known")
    privacy_boundary: str = Field(default="standard", description="隐私边界: strict/standard/relaxed")
    locale: str = Field(default="zh_CN", description="语言 locale")
    resources: Optional[List[Resource]] = Field(default=None, description="RAG 资源列表")


class EnhancePromptRequest(BaseModel):
    prompt: str = Field(..., description="The original prompt to enhance")
    context: Optional[str] = Field(
        "", description="Additional context about the intended use"
    )
    report_style: Optional[str] = Field(
        "bank_business_analysis", description="The style of the report"
    )

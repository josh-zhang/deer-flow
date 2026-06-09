# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import enum
import os

from dotenv import load_dotenv

load_dotenv()

class RAGProvider(enum.Enum):
    DIFY = "dify"
    RAGFLOW = "ragflow"
    VIKINGDB_KNOWLEDGE_BASE = "vikingdb_knowledge_base"
    MOI = "moi"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    MSKB = "mskb"


SELECTED_RAG_PROVIDER = os.getenv("RAG_PROVIDER")

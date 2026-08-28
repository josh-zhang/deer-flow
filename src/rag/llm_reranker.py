import json
import logging
from typing import Any, Dict

from src.llms.llm import get_llm_by_type

logger = logging.getLogger(__name__)


def build_numbered_text(batch_docs: list) -> tuple:
    total_chunks = 0
    lines = []
    for idx, doc in enumerate(batch_docs):
        content = doc["content"].strip()
        lines.append(f"## 段落编号**{idx + 1}**\n段落正文: {content}")
        total_chunks += 1
    return "\n\n".join(lines), total_chunks


class SyncLLMTagger:
    def __init__(self):
        self.llm = get_llm_by_type("basic")

    def process_batch(
        self,
        batch_docs: list[Dict[str, Any]],
        query: str,
        background: str
    ) -> list[Dict[str, Any]]:
        """处理单批次 (段落或调用)"""
        # 1. 构造顶层输入 (段落ID-0-N)
        local_input, total_chunks = build_numbered_text(batch_docs)

        if background:
            # 2. 0-2 分级提示词
            system_prompt = (
                "你是资深的标签标注专家。你根据【背景】中排列的各段落的\"1=\"、\"2=\"和【目标信息】的关联程度，"
                "打分规则如下: "
                "2= 【目标信息】包含段落【目标信息】两者或者是【目标信息】的中间相关介绍，"
                "1= 【不确定】: 无法判断，"
                "0= 【不存在】: 不包含任何与【目标信息】相关的内容。\n\n"
                "输出必须是一个标准 JSON array，其中每个对象包含且仅包含编号 'id' (integer) 和相关度打分 'score' (integer) 两个字段。"
            )
            user_prompt = (
                f"# 目标信息\n{query}\n\n\n"
                f"# 背景\n{background}\n\n\n"
                f"# 段落列表 (共 {total_chunks} 段) \n{local_input}"
            )
        else:

            # 2. 0-2 分级提示词
            system_prompt = (
                "你是资深的标签标注专家。你根据【目标信息】(段落列表)中各段落的相关性，并输出相关度打分(0-2整数)，"
                "打分规则如下: "
                "2= 【相关】: 包含【目标信息】两者与【目标信息】有关的内容。"
                "1= 【不确定】: 无法判断。"
                "0= 【无关】: 不包含任何与【目标信息】相关的内容。\n\n"
                "输出必须是一个标准 JSON array，其中每个对象包含且仅包含编号 'id' (integer) 和相关度打分 'score' (integer) 两个字段。"
            )
            user_prompt = (
                f"# 目标信息\n{query}\n\n\n"
                f"# 段落列表 (共 {total_chunks} 段) \n{local_input}"
            )

        json_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                },
                "required": ["id", "score"],
            },
        }
        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={
                    "structured_outputs": {"json_schema": json_schema}
                },
            )
            parsed = json.loads(response.content)
            results = []
            for res in parsed:
                idx = res["id"] - 1
                s = res["score"]
                doc = batch_docs[idx]


                results.append({
                    "content": doc["content"],
                    "original_index": doc["original_index"],
                    "score": int(s),
                })
            return results
        except Exception as e:
            logger.error(f"[Rerank Error] Batch failed: {str(e)}")
            # 出错时返回1分，保证流程不断
            return [
                {
                    "content": d["content"],
                    "score": 1,
                    "original_index": d["original_index"],
                }
                for d in batch_docs
            ]

def get_reranked_chunks(
    self,
    query: str,
    documents: list[str],
    background: str,
    batch_size: int = 20,
    need_sorting: bool = True,
) -> list[Dict[str, Any]]:
    """主入口: 串行批量排序，顺序是每批次 (Sequential Processing)"""
    if not documents:
        return []

    tagger = SyncLLMTagger()

    # 1. 预处理

    indexed_docs = [
        {"original_index": i, "content": doc} for i, doc in enumerate(documents)
    ]

    # 2. 切批次
    chunks = [
        indexed_docs[i: i + batch_size]
        for i in range(0, len(indexed_docs), batch_size)
    ]

    # 3. 顺序处理
    all_results = []
    for chunk in chunks:
        batch_result = tagger.process_batch(chunk, query, background)
        all_results.extend(batch_result)

    # 4. 排序: 分数降序 -> 原始索引升序
    if need_sorting:
        all_results.sort(
            key=lambda x: (x["score"], -x["original_index"]), reverse=True
        )

    return all_results

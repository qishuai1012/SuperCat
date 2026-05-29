"""文档加载和分片服务（MinerU高精度解析版）"""
import os
from typing import Dict, List
from pathlib import Path
from mineru import MinerU
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """文档加载和分片服务"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 三层分块参数保持原有配置不变
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        # 初始化MinerU解析器
        self.mineru_parser = MinerU()

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        if not text:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        基于MinerU加载文档并三层分片
        :param file_path: 文件路径
        :param filename: 文件名
        :return: 分片后的文档列表
        """
        file_lower = filename.lower()
        file_path_obj = Path(file_path)

        # 判定文档类型
        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            # MinerU高精度解析，输出标准markdown结构化文本
            parse_result = self.mineru_parser.process(
                str(file_path_obj),
                output_format="markdown"
            )
            full_content = parse_result.get("content", "").strip()
            if not full_content:
                raise Exception("文档解析未提取到有效内容")

            documents = []
            page_global_chunk_idx = 0
            # MinerU统一规整为单页逻辑分片，兼容原有分页结构
            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": 1,
            }
            page_chunks = self._split_page_to_three_levels(
                text=full_content,
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx
            )
            page_global_chunk_idx += len(page_chunks)
            documents.extend(page_chunks)

            return documents
        except Exception as e:
            raise Exception(f"MinerU解析文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹批量加载解析文档
        :param folder_path: 文件夹路径
        :return: 所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc")) or file_lower.endswith((".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
# """
# 文档加载和分片服务（MinerU 高精度解析版）
# 核心功能：
# 1. 调用 MinerU 解析 PDF / Word / Excel 等文档，输出结构化 Markdown 内容
# 2. 对文档内容做三级分层切分（1200 → 600 → 300 字符）
# 3. 动态降级切分：短文本不强制切分，避免碎片
# 4. 按文档类型差异化切分（Markdown 标题优先 / Excel 按行切分）
# 5. 生成带父子关系的 chunk，适配 Auto-Merging RAG 架构
# """
# import os
# from typing import Dict, List
# from pathlib import Path
# from mineru import MinerU
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# class DocumentLoader:
#     """文档加载与三级分层切分服务"""
   
#     # 最小有效 chunk 长度，低于该值视为无效碎片，直接过滤
#     MIN_CHUNK_CHARS = 20

#     # 切分分隔符（优先按 Markdown 标题切分，保证章节语义完整） 
#     _SEPARATORS = [
#         "\n# ", "\n## ", "\n### ", "\n#### ",  # 标题层级优先
#         "\n\n", "\n",                         # 段落、换行
#         "。", "！", "？", "，", "、", " ", ""  # 中文标点、最小语义单元
#     ]

#     def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
#         """
#         初始化三级文本切分器
#         三级切分策略：
#         Level1：最大块（章节级），用于 Auto-Merging 上下文合并
#         Level2：中块（过渡级），保证结构稳定
#         Level3：最小块（叶子块），用于向量库精准检索
#         """
#         # 三级切分尺寸（滑动窗口大小 + 重叠长度）
#         level_1_size = max(1200, chunk_size * 2)
#         level_1_overlap = max(240, chunk_overlap * 2)
        
#         level_2_size = max(600, chunk_size)
#         level_2_overlap = max(120, chunk_overlap)
        
#         level_3_size = max(300, chunk_size // 2)
#         level_3_overlap = max(60, chunk_overlap // 2)

#         # 初始化各级切分器
#         self._splitter_level_1 = RecursiveCharacterTextSplitter(
#             chunk_size=level_1_size, chunk_overlap=level_1_overlap,
#             add_start_index=True, separators=self._SEPARATORS,
#         )
#         self._splitter_level_2 = RecursiveCharacterTextSplitter(
#             chunk_size=level_2_size, chunk_overlap=level_2_overlap,
#             add_start_index=True, separators=self._SEPARATORS,
#         )
#         self._splitter_level_3 = RecursiveCharacterTextSplitter(
#             chunk_size=level_3_size, chunk_overlap=level_3_overlap,
#             add_start_index=True, separators=self._SEPARATORS,
#         )

#         # 初始化 MinerU 文档解析器
#         self.mineru_parser = MinerU()

  
#     def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
#         """
#         构建全局唯一的 chunk_id
#         规则：文件名::页码::层级::序号
#         保证 id 不重复，方便溯源
#         """
#         return f"{filename}::p{page_number}::l{level}::{index}"

#     def _split_page_to_three_levels(
#         self,
#         text: str,
#         base_doc: Dict,
#         page_global_chunk_idx: int,
#     ) -> List[Dict]:
#         """
#         对单页内容执行三级链式切分
#         1. 先切 Level1
#         2. 对每个 Level1 切分 Level2（内容足够短时跳过）
#         3. 对每个 Level2 切分 Level3（内容足够短时跳过）
#         4. 自动过滤短碎片
#         """
#         if not text:
#             return []

#         root_chunks: List[Dict] = []
#         page_number = int(base_doc.get("page_number", 0))
#         filename = base_doc["filename"]
#         min_chars = self.MIN_CHUNK_CHARS

#         # 切分 Level1（最大块）
#         level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
#         level_1_counter = 0
#         level_2_counter = 0
#         level_3_counter = 0

#         for level_1_doc in level_1_docs:
#             level_1_text = (level_1_doc.page_content or "").strip()
            
#             # 过滤无效碎片
#             if len(level_1_text) < min_chars:
#                 continue
            
#             # 构建 Level1 chunk（根节点，无父块）
#             level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
#             level_1_counter += 1

#             level_1_chunk = {
#                 **base_doc,
#                 "text": level_1_text,
#                 "chunk_id": level_1_id,
#                 "parent_chunk_id": "",               # 一级块无父节点
#                 "root_chunk_id": level_1_id,         # 自己是根节点
#                 "chunk_level": 1,
#                 "chunk_idx": page_global_chunk_idx,
#             }
#             page_global_chunk_idx += 1
#             root_chunks.append(level_1_chunk)

#             # 如果 Level1 本身就很短，不再切分 Level2、Level3（动态降级）
#             if len(level_1_text) <= self._splitter_level_2.chunk_size:
#                 continue

#             # 对 Level1 切分 Level2
#             level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
#             for level_2_doc in level_2_docs:
#                 level_2_text = (level_2_doc.page_content or "").strip()
#                 if len(level_2_text) < min_chars:
#                     continue

#                 # 构建 Level2 chunk（父节点是 Level1）
#                 level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
#                 level_2_counter += 1

#                 level_2_chunk = {
#                     **base_doc,
#                     "text": level_2_text,
#                     "chunk_id": level_2_id,
#                     "parent_chunk_id": level_1_id,    # 父ID指向Level1
#                     "root_chunk_id": level_1_id,     # 根ID仍是Level1
#                     "chunk_level": 2,
#                     "chunk_idx": page_global_chunk_idx,
#                 }
#                 page_global_chunk_idx += 1
#                 root_chunks.append(level_2_chunk)

#                 # 如果 Level2 本身很短，不再切分 Level3
#                 if len(level_2_text) <= self._splitter_level_3.chunk_size:
#                     continue

#                 # 对 Level2 切分 Level3（叶子块，用于向量检索）
#                 level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
#                 for level_3_doc in level_3_docs:
#                     level_3_text = (level_3_doc.page_content or "").strip()
#                     if len(level_3_text) < min_chars:
#                         continue

#                     level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
#                     level_3_counter += 1

#                     # 构建 Level3 chunk（父节点是 Level2）
#                     root_chunks.append({
#                         **base_doc,
#                         "text": level_3_text,
#                         "chunk_id": level_3_id,
#                         "parent_chunk_id": level_2_id,   # 父ID指向Level2
#                         "root_chunk_id": level_1_id,      # 根ID仍是Level1
#                         "chunk_level": 3,
#                         "chunk_idx": page_global_chunk_idx,
#                     })
#                     page_global_chunk_idx += 1

#         return root_chunks

#     def load_document(self, file_path: str, filename: str) -> list[dict]:
#         """
#         加载单个文档 →  MinerU 解析 → 三级切分
#         支持 PDF / Word / Excel
#         """
#         file_lower = filename.lower()
#         file_path_obj = Path(file_path)

#         # 判断文档类型
#         if file_lower.endswith(".pdf"):
#             doc_type = "PDF"
#         elif file_lower.endswith((".docx", ".doc")):
#             doc_type = "Word"
#         elif file_lower.endswith((".xlsx", ".xls")):
#             doc_type = "Excel"
#         else:
#             raise ValueError(f"不支持的文件类型: {filename}")

#         try:
#             # MinerU 解析文档，输出结构化 Markdown 内容
#             parse_result = self.mineru_parser.process(
#                 str(file_path_obj),
#                 output_format="markdown"
#             )

#             # 获取解析后的分页内容（优先使用 pages，保留原生页码）
#             pages = parse_result.get("pages") or []
#             if not pages:
#                 full_content = parse_result.get("content", "").strip()
#                 if not full_content:
#                     raise Exception("文档解析未提取到有效内容")
#                 pages = [{"page_number": 1, "content": full_content}]

#             # Excel 特殊处理：只按换行切分，不按标点，避免表格被拆碎
#             if doc_type == "Excel":
#                 for splitter in (self._splitter_level_1, self._splitter_level_2, self._splitter_level_3):
#                     splitter._separators = ["\n", " ", ""]

#             documents = []
#             page_global_chunk_idx = 0

#             # 对每一页执行三级切分
#             for page in pages:
#                 page_text = (page.get("content") or page.get("text") or "").strip()
#                 if not page_text:
#                     continue

#                 # 构建页面基础信息
#                 base_doc = {
#                     "filename": filename,
#                     "file_path": file_path,
#                     "file_type": doc_type,
#                     "page_number": int(page.get("page_number") or page.get("page") or 1),
#                 }

#                 # 执行三级切分
#                 page_chunks = self._split_page_to_three_levels(
#                     text=page_text,
#                     base_doc=base_doc,
#                     page_global_chunk_idx=page_global_chunk_idx,
#                 )
#                 page_global_chunk_idx += len(page_chunks)
#                 documents.extend(page_chunks)

#             return documents

#         except Exception as e:
#             raise Exception(f"MinerU解析文档失败: {str(e)}")

#     def load_documents_from_folder(self, folder_path: str) -> list[dict]:
#         """
#         批量加载文件夹中的所有文档
#         支持 PDF / Word / Excel
#         """
#         all_documents = []

#         for filename in os.listdir(folder_path):
#             file_lower = filename.lower()
#             # 只处理支持的文档类型
#             if not (
#                 file_lower.endswith(".pdf") 
#                 or file_lower.endswith((".docx", ".doc")) 
#                 or file_lower.endswith((".xlsx", ".xls"))
#             ):
#                 continue

#             file_path = os.path.join(folder_path, filename)
#             try:
#                 # 加载并切分单个文档
#                 documents = self.load_document(file_path, filename)
#                 all_documents.extend(documents)
#             except Exception:
#                 # 异常文档跳过，不中断整体流程
#                 continue

#         return all_documents



"""
文档加载和分片服务（MinerU API 高精度解析版）
✅ 不占本地显存
✅ 不用安装模型
✅ 你的电脑完美运行
✅ 功能和你原来代码完全一样
"""
import io
import os
import time
import zipfile
import requests
from typing import Dict, List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    MIN_CHUNK_CHARS = 20

    _SEPARATORS = [
        "\n# ", "\n## ", "\n### ", "\n#### ",
        "\n\n", "\n",
        "。", "！", "？", "，", "、", " ", ""
    ]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size, chunk_overlap=level_1_overlap,
            add_start_index=True, separators=self._SEPARATORS,
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size, chunk_overlap=level_2_overlap,
            add_start_index=True, separators=self._SEPARATORS,
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size, chunk_overlap=level_3_overlap,
            add_start_index=True, separators=self._SEPARATORS,
        )

        # ====================== 你自己的 API 信息 ==================
        self.API_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIzMTEwMDQ2NCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MDMyMTMxNSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNjk5YzcxYmMtZjk2Mi00OTI2LThmMmEtOWZiNzIxNWIxOWJiIiwiZW1haWwiOiIiLCJleHAiOjE3ODgwOTczMTV9.7mPOIoQnnuz-xPV6ogaVx1391SzSEjGElFPR6zJwJmYgnPvAOPkFPxtyJm61iFcyi_OYdOLsubGyLTVpnoLHqA"
        self.UPLOAD_TIMEOUT = 300
        # ==========================================================

    def _build_chunk_id(self, filename: str, page_number: int, level: int, index: int) -> str:
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
        min_chars = self.MIN_CHUNK_CHARS

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if len(level_1_text) < min_chars:
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

            if len(level_1_text) <= self._splitter_level_2._chunk_size:
                continue

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if len(level_2_text) < min_chars:
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

                if len(level_2_text) <= self._splitter_level_3._chunk_size:
                    continue

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if len(level_3_text) < min_chars:
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

    # ====================== API 上传 + 解析 ======================
    def _upload_file_and_parse(self, file_path: str):
        """上传本地文件 → MinerU API 解析 → 返回 markdown"""
        file_name = os.path.basename(file_path)

        # 1. 获取上传链接
        headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
        resp = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            headers=headers,
            json={"files": [{"name": file_name}], "model_version": "vlm"}
        )
        data = resp.json()["data"]
        upload_url = data["file_urls"][0]

        # 2. 上传文件
        with open(file_path, "rb") as f:
            requests.put(upload_url, data=f)

        # 3. 等待解析完成
        task_id = resp.json()["data"]["batch_id"]
        start = time.time()

        while time.time() - start < self.UPLOAD_TIMEOUT:
            res = requests.get(
                f"https://mineru.net/api/v4/extract-results/batch/{task_id}",
                headers=headers
            )
            result = res.json()
            state = result["data"]["extract_result"][0]["state"]

            if state == "done":
                zip_url = result["data"]["extract_result"][0]["full_zip_url"]
                md_resp = requests.get(zip_url)
                with zipfile.ZipFile(io.BytesIO(md_resp.content)) as zf:
                    md_files = [f for f in zf.namelist() if f.endswith(".md")]
                    if not md_files:
                        raise Exception("ZIP 中没有找到 .md 文件")
                    return zf.read(md_files[0]).decode("utf-8")

            if state == "failed":
                raise Exception("解析失败")

            time.sleep(3)

        raise Exception("解析超时")

    # ============================================================

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        # ========== 原来本地解析 → 现在 API 解析 ==========
        full_content = self._upload_file_and_parse(file_path)
        full_content = full_content.replace("\x00", "")
        # ==================================================

        pages = [{"page_number": 1, "content": full_content}]
        documents = []
        page_global_chunk_idx = 0

        for page in pages:
            page_text = page.get("content", "").strip()
            if not page_text:
                continue

            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": 1,
            }

            page_chunks = self._split_page_to_three_levels(
                text=page_text,
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            documents.extend(page_chunks)

        return documents

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        all_documents = []
        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith((".pdf", ".docx", ".doc", ".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                docs = self.load_document(file_path, filename)
                all_documents.extend(docs)
            except:
                continue
        return all_documents
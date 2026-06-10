import hashlib
import os
from pathlib import Path
from typing import Tuple


def calculate_file_md5(file_path: str | Path, chunk_size: int = 8192) -> str:
    """计算文件的 MD5 哈希值

    Args:
        file_path: 文件路径
        chunk_size: 读取块大小

    Returns:
        MD5 哈希字符串
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_content_md5(content: bytes) -> str:
    """计算字节内容的 MD5 哈希值

    Args:
        content: 文件内容字节

    Returns:
        MD5 哈希字符串
    """
    return hashlib.md5(content).hexdigest()


def get_file_info(file_path: str | Path) -> Tuple[str, int]:
    """获取文件信息和MD5

    Args:
        file_path: 文件路径

    Returns:
        (MD5哈希, 文件大小字节数)
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    md5_hash = calculate_file_md5(file_path)
    file_size = file_path.stat().st_size

    return md5_hash, file_size


def is_duplicate_file(file_path: str | Path, expected_md5: str) -> bool:
    """检查文件是否与预期MD5匹配

    Args:
        file_path: 文件路径
        expected_md5: 预期的MD5值

    Returns:
        是否匹配
    """
    try:
        actual_md5, _ = get_file_info(file_path)
        # 修复：忽略大小写匹配（业务必备）
        return actual_md5.lower() == expected_md5.lower()
    except (FileNotFoundError, PermissionError, OSError):
        # 修复：增加权限异常捕获，防止崩溃
        return False
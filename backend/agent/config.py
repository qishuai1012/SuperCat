"""
Agent系统配置管理
作用：统一管理智能体（Agent）的所有参数、开关、模型连接信息
属于项目核心配置文件，所有Agent相关功能都会读取这里的设置
"""

# 导入操作系统环境变量工具
import os

# 导入数据类装饰器，用于快速创建配置类（不用写__init__）
from dataclasses import dataclass

# 导入可选类型，允许字段为None
from typing import Optional


# 用@dataclass装饰器，让这个类变成“配置数据容器”，简洁易用
@dataclass
class AgentConfig:
    """
    Agent配置类
    存储：大模型连接信息、并发数、温度、功能开关等所有核心参数
    """

    # --------------------------
    # 大模型连接核心配置（必填）
    # --------------------------
    api_key: str                  # 大模型API密钥（如豆包/通义千问/OpenAI密钥）
    model: str                    # 使用的模型名称（如deepseek-r1/llama3/gpt-4o）
    base_url: str                 # 模型接口地址（私有部署/第三方代理必须填）
    
    # --------------------------
    # 模型生成参数
    # --------------------------
    temperature: float = 0.3       # 温度系数：0=更严谨，1=更发散创意，默认0.3最稳
    
    # --------------------------
    # 并发/性能配置
    # --------------------------
    max_workers: int = 2          # 最大并发工作线程数，控制CPU占用，默认2
    
    # --------------------------
    # 高级功能开关（一键开/关整个模块）
    # --------------------------
    enable_routing: bool = True           # 启用智能路由（简单/复杂问题自动分流）
    enable_decomposition: bool = True     # 启用任务分解（复杂问题拆成小任务）
    enable_reflection: bool = True        # 启用反思Agent（回答后自我检查、修正）
    enable_multi_agent: bool = True       # 启用多智能体协同（多个Agent分工合作）
    enable_parallel: bool = True         # 启用并行执行（同时跑多个子任务）
    enable_cache: bool = True            # 启用智能缓存（加速重复问题）
    enable_learning: bool = True         # 启用学习系统（根据历史对话优化回答）

    @classmethod
    def from_env(cls):
        """
        类方法：从【系统环境变量】中加载配置
        好处：
        1. 不把密钥写死在代码里（安全）
        2. 不同环境（开发/生产）自动切换配置
        """
        return cls(
            # 从环境变量读取密钥
            api_key=os.getenv("ARK_API_KEY"),
            
            # 从环境变量读取模型名称
            model=os.getenv("MODEL"),
            
            # 从环境变量读取接口地址
            base_url=os.getenv("BASE_URL"),
            
            # 其他有默认值的参数（temperature/max_workers/开关）会自动使用默认值
        )
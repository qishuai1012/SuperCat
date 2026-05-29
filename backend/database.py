"""
数据库核心配置文件
作用：
1. 连接 PostgreSQL 数据库
2. 创建 SQLAlchemy 引擎
3. 创建数据库会话（Session）
4. 定义 ORM 模型基类
5. 提供初始化数据库表的方法
"""

# 导入操作系统接口，用于读取环境变量
import os

# 导入 SQLAlchemy 核心工具
# create_engine: 创建数据库连接引擎
from sqlalchemy import create_engine
# declarative_base: 声明 ORM 模型的基类
# sessionmaker: 创建数据库会话工厂
from sqlalchemy.orm import declarative_base, sessionmaker

# ====================== 1. 数据库连接地址 ======================
# 优先从环境变量 DATABASE_URL 读取配置
# 环境变量不存在时，使用默认本地 PostgreSQL 地址
DATABASE_URL = os.getenv(
    "DATABASE_URL",  # 环境变量键名
    # 默认值：postgresql+psycopg2://用户名:密码@主机:端口/数据库名
    "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

# ====================== 2. 创建数据库引擎 ======================
# engine 是 Python 与数据库之间的“通信管道”
# pool_pre_ping=True：每次连接前检查连接是否有效，防止断连
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# ====================== 3. 创建数据库会话工厂 ======================
# SessionLocal 是用来生成“数据库会话（Session）”的工厂
# 每次操作数据库，都要从这里获取一个会话
SessionLocal = sessionmaker(
    bind=engine,         # 绑定到上面创建的数据库引擎
    autoflush=False,     # 关闭自动刷新（手动控制提交）
    autocommit=False,    # 关闭自动提交（手动 commit 才会保存到数据库）
    expire_on_commit=False  # 提交后不销毁对象，方便后续使用
)

# ====================== 4. ORM 模型基类 ======================
# 所有数据库模型（models.py 里的类）都要继承这个 Base
# SQLAlchemy 通过它来识别哪些类需要映射成数据库表
Base = declarative_base()

# ====================== 5. 初始化数据库（创建所有表） ======================
def init_db() -> None:
    """
    初始化数据库
    作用：根据 models 里定义的模型，自动在数据库中创建所有表
    注意：只会创建不存在的表，不会修改/删除已有表
    """
    # 延迟导入 models，避免【循环导入问题】
    import models  # noqa: F401

    # 执行建表操作：让 Base 下所有模型类在数据库中生成对应的表
    Base.metadata.create_all(bind=engine)
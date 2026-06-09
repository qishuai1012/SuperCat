import logging
import os
import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone

# 日志记录器
logger = logging.getLogger(__name__)

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

# 数据库连接与用户模型
from database import SessionLocal
from models import User

# ===================== 安全配置（从环境变量读取，生产环境必须修改）=====================
# JWT 签名密钥
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret")
# JWT 加密算法
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
# 登录令牌过期时间（默认 1440 分钟 = 24 小时）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
# 管理员注册邀请码
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "")
# 密码加密迭代次数（越高越安全）
PBKDF2_ROUNDS = int(os.getenv("PASSWORD_PBKDF2_ROUNDS", "310000"))

# OAuth2 认证配置，指定登录接口地址
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ===================== 数据库连接依赖 =====================
def get_db():
    """
    获取数据库会话
    每个请求自动创建、自动关闭连接
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===================== 密码验证 =====================
def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    验证明文密码与哈希密码是否匹配
    支持：新版 pbkdf2_sha256 + 旧版 bcrypt 兼容
    """
    if not plain_password or not password_hash:
        return False

    # 新版密码格式：pbkdf2_sha256$迭代次数$盐值$哈希值
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            calculated = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                int(rounds),
            )
            # 安全比较，防止计时攻击
            return hmac.compare_digest(calculated, expected)
        except Exception:
            return False

    # 旧版密码格式：bcrypt（兼容历史数据）
    if password_hash.startswith("$2") or password_hash.startswith("$bcrypt"):
        try:
            from passlib.context import CryptContext
            legacy_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")
            return legacy_context.verify(plain_password, password_hash)
        except ImportError:
            logger.warning("passlib 未安装，无法验证 bcrypt 哈希，请安装 passlib[bcrypt]")
            return False
        except Exception as e:
            logger.warning(f"bcrypt 密码验证异常: {e}")
            return False

    return False

# ===================== 密码加密（哈希）=====================
def get_password_hash(password: str) -> str:
    """
    明文密码 → 安全哈希
    算法：PBKDF2-HMAC-SHA256
    输出格式：pbkdf2_sha256$轮次$盐值$摘要
    """
    if not password:
        raise ValueError("password is required")

    # 生成随机盐值
    salt = os.urandom(16)
    # 密码加密
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    # 转为 base64 便于存储
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt_b64}${digest_b64}"

# ===================== 创建 JWT 登录令牌 =====================
def create_access_token(username: str, role: str) -> str:
    """
    生成登录 JWT 令牌
    包含：用户名、角色、过期时间
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,    # 用户名
        "role": role,       # 用户角色
        "exp": expire,      # 过期时间
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ===================== 用户登录验证 =====================
def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """
    登录验证：
    1. 查询用户是否存在
    2. 验证密码是否正确
    成功返回用户对象，失败返回 None
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

# ===================== 获取当前登录用户 =====================
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    从请求头 Token 解析当前登录用户
    所有需要登录的接口都会使用此依赖
    """
    # 无效令牌异常
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的认证令牌",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 解析 Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # 查询用户是否存在
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user

# ===================== 管理员权限校验 =====================
def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    仅管理员可访问
    用于：上传文档、删除文档、管理会话等敏感接口
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员权限不足")
    return current_user

# ===================== 注册时判断用户角色 =====================
def resolve_role(requested_role: str | None, admin_code: str | None) -> str:
    """
    注册时分配角色：
    - 默认普通用户 user
    - 申请 admin 必须提供正确的管理员邀请码
    """
    role = (requested_role or "user").strip().lower()
    if role != "admin":
        return "user"
    if ADMIN_INVITE_CODE and admin_code == ADMIN_INVITE_CODE:
        return "admin"
    raise HTTPException(status_code=403, detail="管理员邀请码错误")
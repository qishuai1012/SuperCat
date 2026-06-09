from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

# 认证相关工具：登录验证、生成Token、密码加密、角色判断
from auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, resolve_role
# 数据库用户模型
from models import User
# 请求/响应格式定义
from schemas import AuthResponse, CurrentUserResponse, LoginRequest, RegisterRequest

# 创建认证模块路由
router = APIRouter()


# ==============================
# 1. 用户注册接口
# ==============================
@router.post("/auth/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,   # 注册请求参数：用户名、密码、角色、管理员码
    db: Session = Depends(get_db)  # 数据库连接
):
    # 去除前后空格，避免无效空格账号
    username = (request.username or "").strip()
    password = (request.password or "").strip()

    # 校验：用户名密码不能为空
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    # 检查用户名是否已被注册
    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    # 解析用户角色：普通用户 / 管理员（需要 admin_code）
    role = resolve_role(request.role, request.admin_code)

    # 创建用户，密码加密存储（不存明文）
    user = User(
        username=username,
        password_hash=get_password_hash(password),
        role=role
    )
    db.add(user)
    db.commit()  # 提交数据库

    # 生成 JWT 登录凭证
    token = create_access_token(username=username, role=role)

    # 返回 Token 和用户信息
    return AuthResponse(access_token=token, username=username, role=role)


# ==============================
# 2. 用户登录接口
# ==============================
@router.post("/auth/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,  # 登录请求：用户名 + 密码
    db: Session = Depends(get_db)
):
    # 验证用户名密码是否正确
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功 → 生成 Token
    token = create_access_token(username=user.username, role=user.role)

    # 返回 Token
    return AuthResponse(access_token=token, username=user.username, role=user.role)


# ==============================
# 3. 获取当前登录用户信息
# ==============================
@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    # 自动验证 Token 是否有效
    # 有效则返回当前登录的用户名和角色
    return CurrentUserResponse(
        username=current_user.username,
        role=current_user.role
    )
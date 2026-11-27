"""JWT authentication and role-based access control.

Identity management decision (REQ-04, #26): the API keeps its own JWT tokens
(HS256, OAuth2 password flow) with two roles, admin and viewer. Microsoft Entra
ID was considered and not adopted for the MVP.

Nothing secret lives in this file. Everything comes from the environment, or
from api/.env (gitignored, copy api/.env.example) when running on the host:

    JWT_SECRET           HMAC key for signing tokens. If unset, a random key is
                         generated at startup, so tokens stop working after a
                         restart. Use at least 32 characters.
    API_ADMIN_PASSWORD   Password for the "admin" account. Unset = no admin login.
    API_VIEWER_PASSWORD  Password for the "viewer" account. Unset = no viewer login.
    JWT_EXPIRE_MINUTES   Token lifetime, default 60.
"""
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Dict, Mapping

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

log = logging.getLogger(__name__)

# Host runs pick up api/.env; real environment variables always win.
load_dotenv(Path(__file__).with_name(".env"), override=False)

ALGORITHM = "HS256"
ROLES = ("admin", "viewer")
MIN_SECRET_LENGTH = 32

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _hash(pw: str) -> bytes:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt())


def load_secret(env: Mapping[str, str]) -> str:
    secret = env.get("JWT_SECRET", "")
    if not secret:
        log.warning("JWT_SECRET is not set; using a random key. Tokens will not survive a restart.")
        return secrets.token_urlsafe(48)
    if len(secret) < MIN_SECRET_LENGTH:
        log.warning("JWT_SECRET is shorter than %d characters.", MIN_SECRET_LENGTH)
    return secret


def load_users(env: Mapping[str, str]) -> Dict[str, dict]:
    """One account per role; the username is the role name."""
    users = {}
    for role in ROLES:
        password = env.get(f"API_{role.upper()}_PASSWORD", "")
        if password:
            users[role] = {"username": role, "hashed_password": _hash(password), "role": role}
        else:
            log.warning("API_%s_PASSWORD is not set; the %s account is disabled.", role.upper(), role)
    return users


def load_expire_minutes(env: Mapping[str, str]) -> int:
    return int(env.get("JWT_EXPIRE_MINUTES", "60"))


SECRET_KEY = load_secret(os.environ)
USERS_DB = load_users(os.environ)
ACCESS_TOKEN_EXPIRE_MINUTES = load_expire_minutes(os.environ)


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


class User(BaseModel):
    username: str
    role: str


def verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


def authenticate_user(username: str, password: str):
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Only HS256 is accepted, so "alg": "none" and RS/HS confusion fail here.
        # jose also rejects an expired "exp" claim.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise credentials_exception
    username = payload.get("sub")
    role = payload.get("role")
    if username is None or role not in ROLES:
        raise credentials_exception
    return User(username=username, role=role)


def require_role(required_role: str):
    """Dependency factory. Admin role implicitly grants access to all roles."""
    async def role_checker(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role != required_role and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}",
            )
        return user
    return role_checker

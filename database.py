import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ==========================
# Database Path
# ==========================

if os.path.exists("/data"):
    DATABASE_URL = "sqlite:////data/prediction.db"
else:
    DATABASE_URL = "sqlite:///./prediction.db"


print(f"[DB] 使用資料庫：{DATABASE_URL}")


# ==========================
# SQLAlchemy Engine
# ==========================

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)


Base = declarative_base()


def get_db():
    """
    FastAPI Database Dependency
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
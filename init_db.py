from database import Base, engine
from models import Prediction


def init_db():
    Base.metadata.create_all(
        bind=engine
    )

    print("✅ SQLite 資料庫與資料表建立完成")


if __name__ == "__main__":
    init_db()
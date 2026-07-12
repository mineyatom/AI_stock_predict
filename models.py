from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)

from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    __table_args__ = (
        UniqueConstraint(
            "predict_date",
            "stock_code",
            name="uq_prediction_date_stock"
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    predict_date = Column(
        String,
        nullable=False,
        index=True
    )

    stock_code = Column(
        String,
        nullable=False,
        index=True
    )

    stock_name = Column(
        String,
        nullable=False
    )

    prediction_text = Column(
        String,
        nullable=False
    )

    confidence = Column(
        Float,
        nullable=False
    )

    up_probability = Column(
        Float,
        nullable=False
    )

    down_probability = Column(
        Float,
        nullable=False
    )

    predict_close = Column(
        Float,
        nullable=False
    )

    lower_price = Column(
        Float,
        nullable=False
    )

    upper_price = Column(
        Float,
        nullable=False
    )

    actual_close = Column(
        Float,
        nullable=True
    )

    actual_change = Column(
        String,
        nullable=True
    )

    is_correct = Column(
        String,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.now,
        nullable=False
    )

    updated_at = Column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False
    )
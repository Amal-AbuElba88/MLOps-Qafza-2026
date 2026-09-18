"""Durable prediction logging for later outcome evaluation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    input_json: Mapped[str] = mapped_column(Text)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    actual_is_late: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class PredictionStore:
    """Store inference records in SQLite locally or PostgreSQL in Compose."""

    def __init__(self, database_url: str, enabled: bool = True):
        self.enabled = enabled
        self.engine = None
        if not enabled:
            return
        if database_url.startswith("sqlite:///"):
            database_path = Path(database_url.removeprefix("sqlite:///"))
            database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    def save(
        self,
        request_id: str,
        payload: Dict[str, Any],
        prediction: int,
        probability: Optional[float],
        model_version: str,
        latency_ms: float,
    ) -> None:
        if not self.enabled or self.engine is None:
            return
        record = PredictionLog(
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
            input_json=json.dumps(payload, default=str, ensure_ascii=False),
            prediction=prediction,
            probability=probability,
            model_version=model_version,
            latency_ms=latency_ms,
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()

"""Thread-safe local SQLite persistence for grading records."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class GradeRecord(Base):
    __tablename__ = "grade_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    speaker: Mapped[str] = mapped_column(String(120), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    overall_score: Mapped[float] = mapped_column(Float)
    feature_vector: Mapped[str] = mapped_column(Text, default="{}")
    coaching_advice: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


StudentAttempt = GradeRecord


class Database:
    def __init__(self, path="presentation_grading.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def save_attempt(self, speaker, filename, report):
        speech = report.get("presentation", {}).get("metrics", {})
        visual = report.get("visual_quality", {}).get("metrics", {})
        gaze = report.get("reading_assessment", {})
        assessment = report.get("communication_assessment", {})
        features = {
            "speech": speech,
            "visual": visual,
            "gaze": gaze,
            "ml_score": report.get("ml_score"),
            "overall_metrics": assessment.get("overall_metrics", {}),
        }
        created_at = datetime.now(timezone.utc)
        with self.session_factory() as session:
            record = GradeRecord(
                speaker=str(speaker or "Default speaker")[:120],
                filename=str(filename or "video")[:255],
                overall_score=float(report.get("overview", {}).get("overall_score", 0) or 0),
                feature_vector=json.dumps(features, ensure_ascii=True),
                coaching_advice=json.dumps(report.get("ai_coaching", {}), ensure_ascii=True),
                created_at=created_at,
            )
            session.add(record)
            session.commit()
            return {"id": record.id, "created_at": created_at.isoformat(timespec="seconds")}

from sqlalchemy import Column, String, Text, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")  # active, completed, paused, failed
    task = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)   # researcher, coder, reviewer, human, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column("metadata", JSON, default=dict)

    session = relationship("Session", back_populates="messages")


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    checkpoint_type = Column(String, nullable=False)  # human_review, cycle_complete, final
    state_snapshot = Column(JSON, nullable=False)
    human_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(String, default="pending")  # pending, approved, rejected, modified

    session = relationship("Session", back_populates="checkpoints")

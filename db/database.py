from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from typing import AsyncGenerator
import os

from .models import Base, Session, Message, Checkpoint

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/sessions.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


class SessionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, task: str, metadata: dict = None) -> Session:
        session = Session(task=task, metadata_=metadata or {})
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get(self, session_id: str) -> Session | None:
        result = await self.db.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def update_status(self, session_id: str, status: str):
        session = await self.get(session_id)
        if session:
            session.status = status
            await self.db.commit()

    async def add_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> Message:
        msg = Message(session_id=session_id, role=role, content=content, metadata_=metadata or {})
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg

    async def add_checkpoint(self, session_id: str, checkpoint_type: str, state: dict) -> Checkpoint:
        cp = Checkpoint(session_id=session_id, checkpoint_type=checkpoint_type, state_snapshot=state)
        self.db.add(cp)
        await self.db.commit()
        await self.db.refresh(cp)
        return cp

    async def resolve_checkpoint(self, checkpoint_id: int, resolution: str, feedback: str = None):
        result = await self.db.execute(select(Checkpoint).where(Checkpoint.id == checkpoint_id))
        cp = result.scalar_one_or_none()
        if cp:
            cp.resolved = resolution
            cp.human_feedback = feedback
            await self.db.commit()

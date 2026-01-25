#!/usr/bin/env python3
"""
Document Store for Session-Scoped Storage

Manages uploaded documents within user sessions.
"""

import uuid
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import logging

from .attachment_handler import ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class SessionDocuments:
    """Documents associated with a session."""
    session_id: str
    created_at: datetime
    documents: Dict[str, ProcessedDocument] = field(default_factory=dict)
    last_access: datetime = field(default_factory=datetime.now)

    def add_document(self, doc_id: str, document: ProcessedDocument):
        """Add a document to the session."""
        self.documents[doc_id] = document
        self.last_access = datetime.now()

    def get_document(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Get a document by ID."""
        self.last_access = datetime.now()
        return self.documents.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the session."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def list_documents(self) -> List[Dict]:
        """List all documents in session."""
        return [
            {
                "id": doc_id,
                "filename": doc.filename,
                "format": doc.format,
                "char_count": len(doc.text),
            }
            for doc_id, doc in self.documents.items()
        ]


class DocumentStore:
    """
    Session-scoped document storage.

    Manages uploaded documents per user session with automatic cleanup.
    """

    DEFAULT_SESSION_TIMEOUT = timedelta(hours=24)
    MAX_DOCUMENTS_PER_SESSION = 10
    CLEANUP_INTERVAL = timedelta(hours=1)

    def __init__(
        self,
        session_timeout: Optional[timedelta] = None,
        max_docs_per_session: int = 10,
    ):
        self.session_timeout = session_timeout or self.DEFAULT_SESSION_TIMEOUT
        self.max_docs_per_session = max_docs_per_session
        self._sessions: Dict[str, SessionDocuments] = {}
        self._lock = threading.Lock()
        self._last_cleanup = datetime.now()

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())

        with self._lock:
            self._sessions[session_id] = SessionDocuments(
                session_id=session_id,
                created_at=datetime.now(),
            )

        self._maybe_cleanup()
        return session_id

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return session_id
        return self.create_session()

    def add_document(
        self,
        session_id: str,
        document: ProcessedDocument,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add a document to a session.

        Returns the document ID.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session not found: {session_id}")

            session = self._sessions[session_id]

            if len(session.documents) >= self.max_docs_per_session:
                raise ValueError(
                    f"Session has reached max documents ({self.max_docs_per_session})"
                )

            doc_id = doc_id or str(uuid.uuid4())[:8]
            session.add_document(doc_id, document)

        return doc_id

    def get_document(
        self,
        session_id: str,
        doc_id: str,
    ) -> Optional[ProcessedDocument]:
        """Get a document from a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.get_document(doc_id)
        return None

    def remove_document(self, session_id: str, doc_id: str) -> bool:
        """Remove a document from a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.remove_document(doc_id)
        return False

    def list_documents(self, session_id: str) -> List[Dict]:
        """List all documents in a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.list_documents()
        return []

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            return {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_access": session.last_access.isoformat(),
                "document_count": len(session.documents),
                "total_chars": sum(len(d.text) for d in session.documents.values()),
            }

    def clear_session(self, session_id: str):
        """Clear all documents from a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].documents.clear()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed."""
        now = datetime.now()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return

        self._cleanup_expired_sessions()
        self._last_cleanup = now

    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_access > self.session_timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

    def get_global_stats(self) -> Dict:
        """Get global store statistics."""
        with self._lock:
            total_docs = sum(len(s.documents) for s in self._sessions.values())
            total_chars = sum(
                sum(len(d.text) for d in s.documents.values())
                for s in self._sessions.values()
            )

            return {
                "active_sessions": len(self._sessions),
                "total_documents": total_docs,
                "total_characters": total_chars,
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            }


# Global document store instance
document_store = DocumentStore()


def main():
    """Demo of document store."""
    store = DocumentStore()

    # Create session
    session_id = store.create_session()
    print(f"Created session: {session_id}")

    # Show stats
    stats = store.get_global_stats()
    print(f"Global stats: {stats}")


if __name__ == "__main__":
    main()

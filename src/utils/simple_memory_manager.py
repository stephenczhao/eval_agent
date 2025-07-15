"""
Tennis Intelligence System - Simple Memory Manager
==================================================

Simplified memory manager that stores only the last 5 conversation pairs
without semantic similarity - just simple append and retrieve.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    from src.models.classifier_models import ConversationPair
except ImportError:
    from models.classifier_models import ConversationPair


@dataclass
class SimpleMemoryEntry:
    """Simple memory entry with just user query and system response."""
    timestamp: datetime
    user_query: str
    system_response: str
    session_id: str


class SimpleMemoryManager:
    """
    Simplified memory manager that stores only the last 5 conversation pairs
    per session without any semantic similarity or complex retrieval.
    
    This is optimized for efficient context passing to agents.
    """
    
    def __init__(self, memory_db_path: str = "tennis_data/memory.db"):
        """Initialize simple memory manager."""
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(exist_ok=True)
        self._init_simple_database()
    
    def _init_simple_database(self) -> None:
        """Initialize simple database with just conversation history."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Simple conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simple_conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_query TEXT NOT NULL,
                system_response TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for efficient retrieval
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp 
            ON simple_conversation_history(session_id, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def store_conversation(
        self,
        session_id: str,
        user_query: str,
        system_response: str
    ) -> None:
        """Store a conversation pair."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            # Store the conversation
            cursor.execute("""
                INSERT INTO simple_conversation_history 
                (session_id, timestamp, user_query, system_response)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(),
                user_query,
                system_response[:1000]  # Truncate long responses
            ))
            
            # Keep only last 10 entries per session (we'll use last 5)
            cursor.execute("""
                DELETE FROM simple_conversation_history 
                WHERE session_id = ? 
                AND id NOT IN (
                    SELECT id FROM simple_conversation_history 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                )
            """, (session_id, session_id))
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Simple memory storage failed: {e}")
        finally:
            conn.close()
    
    def get_conversation_history(
        self,
        session_id: str,
        max_pairs: int = 5
    ) -> List[ConversationPair]:
        """Get the last N conversation pairs for a session."""
        if not session_id:
            return []
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT user_query, system_response
                FROM simple_conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, max_pairs))
            
            results = cursor.fetchall()
            
            # Return in chronological order (oldest first)
            conversation_pairs = []
            for row in reversed(results):
                conversation_pairs.append(ConversationPair(
                    user_query=row[0],
                    system_response=row[1]
                ))
            
            return conversation_pairs
            
        except Exception as e:
            print(f"⚠️  Simple memory retrieval failed: {e}")
            return []
        finally:
            conn.close()
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM simple_conversation_history 
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Session clear failed: {e}")
        finally:
            conn.close()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get basic stats for a session."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM simple_conversation_history
                WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            
            return {
                "total_conversations": result[0] if result else 0,
                "first_interaction": result[1] if result and result[1] else None,
                "last_interaction": result[2] if result and result[2] else None
            }
            
        except Exception as e:
            print(f"⚠️  Session stats failed: {e}")
            return {"total_conversations": 0}
        finally:
            conn.close()
    
    def cleanup_old_sessions(self, days_to_keep: int = 7) -> None:
        """Clean up old conversation history."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM simple_conversation_history 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Memory cleanup failed: {e}")
        finally:
            conn.close()


def create_session_id() -> str:
    """Create a unique session ID."""
    import uuid
    return str(uuid.uuid4())[:8] 
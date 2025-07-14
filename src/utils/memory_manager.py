"""
Tennis Intelligence System - Memory Management
==============================================

Manages conversation history, context persistence, and user interaction memory
for the tennis intelligence system.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ConversationMemoryEntry:
    """Structure for conversation memory entries."""
    timestamp: datetime
    user_query: str
    system_response: str
    sources_used: List[str]
    confidence_score: float
    execution_time: float


@dataclass  
class EntityExtraction:
    """Structure for extracted tennis entities."""
    entity_type: str  # 'player', 'tournament', 'surface', 'year', etc.
    entity_value: str
    confidence: float
    context: str  # Where in the query it was found


@dataclass
class UserSession:
    """Represents a user session with preferences and context."""
    session_id: str
    start_time: datetime
    last_activity: datetime
    user_preferences: Dict[str, Any]
    conversation_count: int
    favorite_players: List[str]
    preferred_topics: List[str]


class MemoryManager:
    """
    Manages conversation memory and user context for the tennis intelligence system.
    
    Features:
    - Persistent conversation history
    - User preference tracking
    - Context extraction and retention
    - Entity memory (players, tournaments mentioned)
    - Session management
    """
    
    def __init__(self, memory_db_path: str = "tennis_data/memory.db"):
        """
        Initialize the memory manager.
        
        Args:
            memory_db_path: Path to SQLite database for memory storage
        """
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(exist_ok=True)
        self._init_memory_database()
    
    def _init_memory_database(self) -> None:
        """Initialize the memory database with required tables."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_query TEXT NOT NULL,
                system_response TEXT NOT NULL,
                sources_used TEXT,  -- JSON array
                confidence_score REAL,
                execution_time REAL,
                tennis_entities TEXT,  -- JSON array
                query_intent TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME NOT NULL,
                last_activity DATETIME NOT NULL,
                user_preferences TEXT,  -- JSON object
                conversation_count INTEGER DEFAULT 0,
                favorite_players TEXT,  -- JSON array
                preferred_topics TEXT,  -- JSON array
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Entity memory table (tracks mentioned players, tournaments, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                first_mentioned DATETIME NOT NULL,
                last_mentioned DATETIME NOT NULL,
                mention_count INTEGER DEFAULT 1,
                context TEXT,  -- Where/how it was mentioned
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Query patterns table (for learning user preferences)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query_type TEXT NOT NULL,  -- statistical, current_events, etc.
                frequency INTEGER DEFAULT 1,
                last_used DATETIME NOT NULL,
                success_rate REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_conversation(
        self,
        session_id: str,
        user_query: str,
        system_response: str,
        sources_used: List[str] = None,
        confidence_score: float = 0.0,
        execution_time: float = 0.0,
        tennis_entities: List[str] = None,
        query_intent: str = "unknown"
    ) -> None:
        """Store a conversation turn in memory - simplified to just user query and response."""
        conn = None
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Simple conversation storage - just user query and system response
            cursor.execute("""
                INSERT INTO conversation_history 
                (session_id, timestamp, user_query, system_response, sources_used,
                 confidence_score, execution_time, tennis_entities, query_intent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now(),
                user_query,
                system_response,
                json.dumps(sources_used or []),
                confidence_score,
                execution_time,
                json.dumps(tennis_entities or []),
                query_intent
            ))
            
            conn.commit()
            
        except sqlite3.OperationalError as e:
            # Handle database lock gracefully
            print(f"⚠️  Memory storage skipped due to database lock: {e}")
        except Exception as e:
            print(f"⚠️  Memory storage failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[ConversationMemoryEntry]:
        """Get recent conversation history for a session."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, user_query, system_response, sources_used,
                   confidence_score, execution_time
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            ConversationMemoryEntry(
                timestamp=datetime.fromisoformat(row[0]),
                user_query=row[1],
                system_response=row[2],
                sources_used=json.loads(row[3]) if row[3] else [],
                confidence_score=row[4],
                execution_time=row[5]
            )
            for row in results
        ]
    
    def get_relevant_context(
        self,
        session_id: str,
        current_query: str,
        max_entries: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get contextually relevant conversation history for the current query.
        
        Uses simple keyword matching to find relevant past conversations.
        """
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Get recent conversations with keyword overlap
        query_words = set(current_query.lower().split())
        
        cursor.execute("""
            SELECT timestamp, user_query, system_response, tennis_entities, confidence_score
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """, (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Score relevance based on keyword overlap
        relevant_contexts = []
        for row in results:
            past_query = row[1].lower()
            past_words = set(past_query.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(past_words))
            if overlap > 0:
                # Also check entity overlap
                entities = json.loads(row[3]) if row[3] else []
                entity_overlap = len([e for e in entities if e.lower() in current_query.lower()])
                
                relevance_score = overlap + (entity_overlap * 2)  # Weight entities higher
                
                relevant_contexts.append({
                    "timestamp": row[0],
                    "user_query": row[1],
                    "system_response": row[2],
                    "entities": entities,
                    "confidence_score": row[4],
                    "relevance_score": relevance_score
                })
        
        # Sort by relevance and return top results
        relevant_contexts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_contexts[:max_entries]
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences and patterns for personalization."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("""
            SELECT user_preferences, favorite_players, preferred_topics
            FROM user_sessions
            WHERE session_id = ?
        """, (session_id,))
        
        session_result = cursor.fetchone()
        
        # Get query patterns
        cursor.execute("""
            SELECT query_type, frequency, success_rate
            FROM query_patterns
            WHERE session_id = ?
            ORDER BY frequency DESC
        """, (session_id,))
        
        pattern_results = cursor.fetchall()
        
        # Get frequently mentioned entities
        cursor.execute("""
            SELECT entity_type, entity_value, mention_count
            FROM entity_memory
            WHERE session_id = ?
            ORDER BY mention_count DESC
            LIMIT 10
        """, (session_id,))
        
        entity_results = cursor.fetchall()
        
        conn.close()
        
        preferences = {
            "user_preferences": json.loads(session_result[0]) if session_result and session_result[0] else {},
            "favorite_players": json.loads(session_result[1]) if session_result and session_result[1] else [],
            "preferred_topics": json.loads(session_result[2]) if session_result and session_result[2] else [],
            "query_patterns": [
                {"type": row[0], "frequency": row[1], "success_rate": row[2]}
                for row in pattern_results
            ],
            "frequent_entities": [
                {"type": row[0], "value": row[1], "mentions": row[2]}
                for row in entity_results
            ]
        }
        
        return preferences
    
    def _update_session_activity(self, session_id: str) -> None:
        """Update session activity timestamp."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Update existing session or create new one
        cursor.execute("""
            INSERT OR REPLACE INTO user_sessions 
            (session_id, start_time, last_activity, conversation_count)
            VALUES (
                ?,
                COALESCE((SELECT start_time FROM user_sessions WHERE session_id = ?), ?),
                ?,
                COALESCE((SELECT conversation_count FROM user_sessions WHERE session_id = ?), 0) + 1
            )
        """, (
            session_id, session_id, datetime.now(),
            datetime.now(), session_id
        ))
        
        conn.commit()
        conn.close()
    
    def _update_entity_memory(self, session_id: str, entities: List[str], context: str) -> None:
        """Update entity memory with new mentions."""
        if not entities:
            return
            
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        for entity in entities:
            # Determine entity type (simple heuristic)
            entity_type = self._classify_entity_type(entity)
            
            # Update or insert entity
            cursor.execute("""
                INSERT OR REPLACE INTO entity_memory
                (session_id, entity_type, entity_value, first_mentioned, last_mentioned, 
                 mention_count, context)
                VALUES (
                    ?, ?, ?,
                    COALESCE((SELECT first_mentioned FROM entity_memory WHERE session_id = ? AND entity_value = ?), ?),
                    ?,
                    COALESCE((SELECT mention_count FROM entity_memory WHERE session_id = ? AND entity_value = ?), 0) + 1,
                    ?
                )
            """, (
                session_id, entity_type, entity,
                session_id, entity, datetime.now(),
                datetime.now(),
                session_id, entity,
                context[:200]  # Truncate context
            ))
        
        conn.commit()
        conn.close()
    
    def _update_query_patterns(self, session_id: str, query_intent: str) -> None:
        """Update query pattern frequency."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO query_patterns
            (session_id, query_type, frequency, last_used)
            VALUES (
                ?, ?,
                COALESCE((SELECT frequency FROM query_patterns WHERE session_id = ? AND query_type = ?), 0) + 1,
                ?
            )
        """, (session_id, query_intent, session_id, query_intent, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def _classify_entity_type(self, entity: str) -> str:
        """Simple entity type classification."""
        entity_lower = entity.lower()
        
        # Tennis tournaments
        tournaments = ['open', 'cup', 'masters', 'wimbledon', 'roland garros', 'us open', 'australian']
        if any(term in entity_lower for term in tournaments):
            return 'tournament'
        
        # Surfaces
        surfaces = ['clay', 'hard', 'grass', 'carpet']
        if entity_lower in surfaces:
            return 'surface'
        
        # Years
        if entity.isdigit() and 1950 <= int(entity) <= 2030:
            return 'year'
        
        # Default to player
        return 'player'
    
    def cleanup_old_memory(self, days_to_keep: int = 30) -> None:
        """Clean up old conversation history to manage database size."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversation_history 
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        cursor.execute("""
            DELETE FROM entity_memory 
            WHERE last_mentioned < ?
        """, (cutoff_date,))
        
        conn.commit()
        conn.close()


def create_session_id() -> str:
    """Create a unique session ID."""
    import uuid
    return str(uuid.uuid4())[:8] 
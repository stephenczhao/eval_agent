"""
Enhanced Memory Manager with Semantic Context Awareness
======================================================

Replaces keyword-based memory retrieval with semantic similarity for better context understanding.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


@dataclass
class ContextEntry:
    """Enhanced context entry with semantic information."""
    timestamp: datetime
    user_query: str
    system_response: str
    entities: List[str]
    intent: str
    confidence: float
    embedding: Optional[np.ndarray] = None
    relevance_score: float = 0.0


class EnhancedMemoryManager:
    """
    Enhanced memory manager with semantic context awareness.
    
    Improvements:
    - Semantic similarity instead of keyword matching
    - Context compression for long conversations
    - Intelligent entity relationship mapping
    - Performance-optimized retrieval
    """
    
    def __init__(self, memory_db_path: str = "tennis_data/memory.db"):
        """Initialize enhanced memory manager."""
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(exist_ok=True)
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Cache for embeddings and frequent queries
        self.embedding_cache = {}
        self.query_cache = {}
        
        self._init_enhanced_database()
        self._load_or_create_vectorizer()
    
    def _init_enhanced_database(self) -> None:
        """Initialize enhanced database with semantic features."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        # Enhanced conversation history with semantic features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_query TEXT NOT NULL,
                system_response TEXT NOT NULL,
                entities TEXT,  -- JSON array
                intent TEXT,
                confidence_score REAL,
                query_embedding BLOB,  -- Serialized numpy array
                context_tokens TEXT,  -- Key context tokens
                relevance_decay REAL DEFAULT 1.0,  -- Decay factor for relevance
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexes for performance
                INDEX idx_session_timestamp (session_id, timestamp),
                INDEX idx_intent (intent),
                INDEX idx_confidence (confidence_score)
            )
        """)
        
        # Entity relationship mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1 TEXT NOT NULL,
                entity2 TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                context_count INTEGER DEFAULT 1,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(entity1, entity2, relationship_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_or_create_vectorizer(self) -> None:
        """Load existing vectorizer or create new one."""
        vectorizer_path = self.memory_db_path.parent / "vectorizer.pkl"
        
        if vectorizer_path.exists():
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load vectorizer: {e}, creating new one")
                self._fit_vectorizer_on_existing_data()
        else:
            self._fit_vectorizer_on_existing_data()
    
    def _fit_vectorizer_on_existing_data(self) -> None:
        """Fit vectorizer on existing conversation data."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_query, system_response 
            FROM enhanced_conversation_history 
            ORDER BY timestamp DESC 
            LIMIT 1000
        """)
        
        texts = []
        for row in cursor.fetchall():
            texts.extend([row[0], row[1]])
        
        conn.close()
        
        if texts:
            try:
                self.vectorizer.fit(texts)
                self._save_vectorizer()
            except Exception as e:
                print(f"⚠️  Failed to fit vectorizer: {e}")
    
    def _save_vectorizer(self) -> None:
        """Save vectorizer to disk."""
        vectorizer_path = self.memory_db_path.parent / "vectorizer.pkl"
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        except Exception as e:
            print(f"⚠️  Failed to save vectorizer: {e}")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get semantic embedding for query."""
        if query in self.embedding_cache:
            return self.embedding_cache[query]
        
        try:
            embedding = self.vectorizer.transform([query]).toarray()[0]
            self.embedding_cache[query] = embedding
            return embedding
        except Exception as e:
            print(f"⚠️  Failed to get embedding: {e}")
            return np.zeros(1000)  # Fallback to zero vector
    
    def store_conversation_enhanced(
        self,
        session_id: str,
        user_query: str,
        system_response: str,
        entities: List[str] = None,
        intent: str = "unknown",
        confidence: float = 0.0
    ) -> None:
        """Store conversation with enhanced semantic features."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            # Get query embedding
            embedding = self._get_query_embedding(user_query)
            embedding_blob = embedding.tobytes()
            
            # Extract context tokens
            context_tokens = self._extract_context_tokens(user_query, system_response)
            
            cursor.execute("""
                INSERT INTO enhanced_conversation_history 
                (session_id, timestamp, user_query, system_response, entities, 
                 intent, confidence_score, query_embedding, context_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(),
                user_query,
                system_response[:2000],  # Truncate long responses
                json.dumps(entities or []),
                intent,
                confidence,
                embedding_blob,
                json.dumps(context_tokens)
            ))
            
            # Update entity relationships
            if entities:
                self._update_entity_relationships(entities)
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Enhanced memory storage failed: {e}")
        finally:
            conn.close()
    
    def _extract_context_tokens(self, query: str, response: str) -> List[str]:
        """Extract key context tokens from query and response."""
        try:
            # Combine query and response
            combined_text = f"{query} {response}"
            
            # Get TF-IDF features
            tfidf_matrix = self.vectorizer.transform([combined_text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top tokens
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[-20:]  # Top 20 tokens
            
            return [feature_names[i] for i in top_indices if scores[i] > 0]
            
        except Exception as e:
            print(f"⚠️  Context token extraction failed: {e}")
            return []
    
    def _update_entity_relationships(self, entities: List[str]) -> None:
        """Update entity relationship mapping."""
        if len(entities) < 2:
            return
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            # Create relationships between entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    cursor.execute("""
                        INSERT OR REPLACE INTO entity_relationships 
                        (entity1, entity2, relationship_type, confidence, context_count, last_seen)
                        VALUES (?, ?, ?, ?, 
                                COALESCE((SELECT context_count FROM entity_relationships 
                                         WHERE entity1 = ? AND entity2 = ?), 0) + 1, ?)
                    """, (
                        entity1, entity2, "co_occurrence", 0.8,
                        entity1, entity2, datetime.now()
                    ))
            
            conn.commit()
            
        except Exception as e:
            print(f"⚠️  Entity relationship update failed: {e}")
        finally:
            conn.close()
    
    def get_semantic_context(
        self,
        session_id: str,
        current_query: str,
        max_entries: int = 5,
        similarity_threshold: float = 0.1
    ) -> List[ContextEntry]:
        """Get contextually relevant conversation history using semantic similarity."""
        if not current_query.strip():
            return []
        
        # Check cache first
        cache_key = f"{session_id}_{current_query}_{max_entries}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            # Get recent conversation history
            cursor.execute("""
                SELECT timestamp, user_query, system_response, entities, 
                       intent, confidence_score, query_embedding, context_tokens
                FROM enhanced_conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (session_id,))
            
            results = cursor.fetchall()
            
            if not results:
                return []
            
            # Get current query embedding
            current_embedding = self._get_query_embedding(current_query)
            
            # Calculate semantic similarity
            context_entries = []
            for row in results:
                try:
                    # Reconstruct embedding
                    if row[6]:  # query_embedding
                        stored_embedding = np.frombuffer(row[6], dtype=np.float64)
                        if stored_embedding.shape == current_embedding.shape:
                            similarity = cosine_similarity(
                                current_embedding.reshape(1, -1),
                                stored_embedding.reshape(1, -1)
                            )[0][0]
                        else:
                            similarity = 0.0
                    else:
                        similarity = 0.0
                    
                    # Apply time decay
                    timestamp = datetime.fromisoformat(row[0])
                    time_diff = (datetime.now() - timestamp).total_seconds()
                    time_decay = np.exp(-time_diff / 86400)  # 1 day decay constant
                    
                    final_score = similarity * time_decay
                    
                    if final_score >= similarity_threshold:
                        context_entries.append(ContextEntry(
                            timestamp=timestamp,
                            user_query=row[1],
                            system_response=row[2],
                            entities=json.loads(row[3]) if row[3] else [],
                            intent=row[4],
                            confidence=row[5],
                            relevance_score=final_score
                        ))
                        
                except Exception as e:
                    print(f"⚠️  Error processing context entry: {e}")
                    continue
            
            # Sort by relevance and return top entries
            context_entries.sort(key=lambda x: x.relevance_score, reverse=True)
            result = context_entries[:max_entries]
            
            # Cache the result
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"⚠️  Semantic context retrieval failed: {e}")
            return []
        finally:
            conn.close()
    
    def get_entity_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """Get related entities for better context understanding."""
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT entity2, relationship_type, confidence, context_count
                FROM entity_relationships
                WHERE entity1 = ?
                ORDER BY confidence DESC, context_count DESC
                LIMIT 10
            """, (entity,))
            
            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    "related_entity": row[0],
                    "relationship_type": row[1],
                    "confidence": row[2],
                    "context_count": row[3]
                })
            
            return relationships
            
        except Exception as e:
            print(f"⚠️  Entity relationship retrieval failed: {e}")
            return []
        finally:
            conn.close()
    
    def cleanup_memory_intelligent(self, days_to_keep: int = 30, min_relevance: float = 0.1) -> None:
        """Intelligent memory cleanup based on relevance and time."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()
        
        try:
            # Remove old, low-relevance entries
            cursor.execute("""
                DELETE FROM enhanced_conversation_history 
                WHERE timestamp < ? AND confidence_score < ?
            """, (cutoff_date, min_relevance))
            
            # Clean up entity relationships
            cursor.execute("""
                DELETE FROM entity_relationships 
                WHERE last_seen < ? AND confidence < ?
            """, (cutoff_date, min_relevance))
            
            conn.commit()
            
            # Clear caches
            self.embedding_cache.clear()
            self.query_cache.clear()
            
        except Exception as e:
            print(f"⚠️  Intelligent memory cleanup failed: {e}")
        finally:
            conn.close() 
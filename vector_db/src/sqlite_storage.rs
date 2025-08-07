use crate::{vector::Vector, Result};
use rusqlite::{Connection, Result as SqliteResult, params, Row};
use uuid::Uuid;
use std::path::Path;
use std::collections::HashMap;
use serde_json::Value;

pub struct SQLiteStorage {
    conn: Connection,
    collection_name: String,
}

impl SQLiteStorage {
    pub fn new<P: AsRef<Path>>(db_path: P, collection_name: &str) -> Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to open SQLite database: {}", e)))?;
        
        let storage = Self {
            conn,
            collection_name: collection_name.to_string(),
        };
        
        storage.init_tables()?;
        Ok(storage)
    }

    fn init_tables(&self) -> Result<()> {
        // Create collections table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to create collections table: {}", e)))?;

        // Create vectors table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                collection_id INTEGER NOT NULL,
                dimension INTEGER NOT NULL,
                data BLOB NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_id) REFERENCES collections (id)
            )",
            [],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to create vectors table: {}", e)))?;

        // Create system_info table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS system_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to create system_info table: {}", e)))?;

        // Insert or update collection
        self.conn.execute(
            "INSERT OR REPLACE INTO collections (name, updated_at) VALUES (?, CURRENT_TIMESTAMP)",
            params![self.collection_name],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to insert collection: {}", e)))?;

        Ok(())
    }

    pub fn insert_vector(&self, vector: &Vector) -> Result<()> {
        let collection_id = self.get_collection_id()?;
        let data = bincode::serialize(&vector.data)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize vector data: {}", e)))?;
        
        let metadata = vector.metadata.as_ref()
            .map(|m| serde_json::to_string(m))
            .transpose()
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize metadata: {}", e)))?;

        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (id, collection_id, dimension, data, metadata) VALUES (?, ?, ?, ?, ?)",
            params![
                vector.id.to_string(),
                collection_id,
                vector.dimension(),
                data,
                metadata
            ],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to insert vector: {}", e)))?;

        Ok(())
    }

    pub fn get_vector(&self, id: &Uuid) -> Result<Option<Vector>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, dimension, data, metadata FROM vectors WHERE id = ?"
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to prepare query: {}", e)))?;

        let mut rows = stmt.query(params![id.to_string()])
            .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to execute query: {}", e)))?;

        if let Some(row) = rows.next()
            .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to fetch row: {}", e)))? {
            let vector = self.row_to_vector(&row)?;
            Ok(Some(vector))
        } else {
            Ok(None)
        }
    }

    pub fn delete_vector(&self, id: &Uuid) -> Result<()> {
        self.conn.execute(
            "DELETE FROM vectors WHERE id = ?",
            params![id.to_string()],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    pub fn get_all_vectors(&self) -> Result<Vec<Vector>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, dimension, data, metadata FROM vectors ORDER BY created_at"
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt.query([])
            .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to execute query: {}", e)))?;

        let mut vectors = Vec::new();
        for row in rows {
            let row = row.map_err(|e| crate::VectorDBError::StorageError(format!("Failed to fetch row: {}", e)))?;
            let vector = self.row_to_vector(&row)?;
            vectors.push(vector);
        }

        Ok(vectors)
    }

    pub fn count_vectors(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vectors",
            [],
            |row| row.get(0),
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to count vectors: {}", e)))?;

        Ok(count as usize)
    }

    pub fn set_system_info(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO system_info (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            params![key, value],
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to set system info: {}", e)))?;

        Ok(())
    }

    pub fn get_system_info(&self, key: &str) -> Result<Option<String>> {
        let value: Option<String> = self.conn.query_row(
            "SELECT value FROM system_info WHERE key = ?",
            params![key],
            |row| row.get(0),
        ).optional()
        .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to get system info: {}", e)))?;

        Ok(value)
    }

    fn get_collection_id(&self) -> Result<i64> {
        let id: i64 = self.conn.query_row(
            "SELECT id FROM collections WHERE name = ?",
            params![self.collection_name],
            |row| row.get(0),
        ).map_err(|e| crate::VectorDBError::StorageError(format!("Failed to get collection ID: {}", e)))?;

        Ok(id)
    }

    fn row_to_vector(&self, row: &Row) -> Result<Vector> {
        let id_str: String = row.get(0)?;
        let dimension: i64 = row.get(1)?;
        let data_blob: Vec<u8> = row.get(2)?;
        let metadata_str: Option<String> = row.get(3)?;

        let id = Uuid::parse_str(&id_str)
            .map_err(|e| crate::VectorDBError::StorageError(format!("Failed to parse UUID: {}", e)))?;

        let data = bincode::deserialize(&data_blob)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize vector data: {}", e)))?;

        let metadata = if let Some(metadata_str) = metadata_str {
            Some(serde_json::from_str(&metadata_str)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize metadata: {}", e)))?)
        } else {
            None
        };

        Ok(Vector {
            id,
            data,
            metadata,
        })
    }
} 
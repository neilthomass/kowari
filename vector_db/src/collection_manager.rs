use crate::{vector::Vector, Result};
use std::path::Path;
use uuid::Uuid;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::sqlite_storage::SQLiteStorage;
use super::binary_index::BinaryIndex;

fn get_current_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string()
}

pub struct CollectionManager {
    base_path: std::path::PathBuf,
    collections: HashMap<String, Collection>,
}

pub struct Collection {
    name: String,
    pub sqlite_storage: SQLiteStorage,
    binary_index: BinaryIndex,
    dimension: usize,
}

impl CollectionManager {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create base directory: {}", e)))?;

        Ok(Self {
            base_path,
            collections: HashMap::new(),
        })
    }

    pub fn create_collection(&mut self, name: &str, dimension: usize) -> Result<()> {
        let collection_path = self.base_path.join(name);
        std::fs::create_dir_all(&collection_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create collection directory: {}", e)))?;

        // Create SQLite database for metadata
        let db_path = collection_path.join("metadata.sqlite3");
        let sqlite_storage = SQLiteStorage::new(&db_path, name)?;

        // Create binary index file for vectors
        let index_path = collection_path.join("vectors.kwi");
        let binary_index = BinaryIndex::new(&index_path, dimension)?;

        let collection = Collection {
            name: name.to_string(),
            sqlite_storage,
            binary_index,
            dimension,
        };

        self.collections.insert(name.to_string(), collection);

        // Set system info
        self.set_system_info(name, "dimension", &dimension.to_string())?;
        self.set_system_info(name, "created_at", &get_current_timestamp())?;

        Ok(())
    }

    pub fn get_collection(&mut self, name: &str) -> Result<Option<&mut Collection>> {
        if !self.collections.contains_key(name) {
            // Try to load existing collection
            self.load_collection(name)?;
        }

        Ok(self.collections.get_mut(name))
    }

    fn load_collection(&mut self, name: &str) -> Result<()> {
        let collection_path = self.base_path.join(name);
        if !collection_path.exists() {
            return Ok(());
        }

        let db_path = collection_path.join("metadata.sqlite3");
        let index_path = collection_path.join("vectors.kwi");

        if !db_path.exists() || !index_path.exists() {
            return Ok(());
        }

        let sqlite_storage = SQLiteStorage::new(&db_path, name)?;
        let binary_index = BinaryIndex::new(&index_path, 128)?; // Default dimension, will be updated
        let dimension = binary_index.get_dimension();

        let collection = Collection {
            name: name.to_string(),
            sqlite_storage,
            binary_index,
            dimension,
        };

        self.collections.insert(name.to_string(), collection);

        Ok(())
    }

    pub fn list_collections(&self) -> Result<Vec<String>> {
        let mut collections = Vec::new();
        
        for entry in std::fs::read_dir(&self.base_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read base directory: {}", e)))? {
            let entry = entry
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read directory entry: {}", e)))?;
            
            if entry.file_type()?.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if self.base_path.join(&name).join("metadata.sqlite3").exists() {
                    collections.push(name);
                }
            }
        }

        Ok(collections)
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        if let Some(_) = self.collections.remove(name) {
            let collection_path = self.base_path.join(name);
            if collection_path.exists() {
                std::fs::remove_dir_all(&collection_path)
                    .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to delete collection directory: {}", e)))?;
            }
        }

        Ok(())
    }

    pub fn add_vector(&mut self, collection_name: &str, vector: &Vector) -> Result<()> {
        let collection = self.get_collection(collection_name)?
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        // Validate dimension
        if vector.dimension() != collection.dimension {
            return Err(crate::VectorDBError::StorageError(
                format!("Vector dimension {} doesn't match collection dimension {}", 
                    vector.dimension(), collection.dimension)
            ));
        }

        // Store in SQLite for metadata and system info
        collection.sqlite_storage.insert_vector(vector)?;

        // Store in binary index for fast retrieval
        collection.binary_index.add_vector(vector)?;

        // Update system info
        let count = collection.binary_index.count_vectors();
        self.set_system_info(collection_name, "vector_count", &count.to_string())?;
        self.set_system_info(collection_name, "updated_at", &get_current_timestamp())?;

        Ok(())
    }

    pub fn get_vector(&self, collection_name: &str, id: &Uuid) -> Result<Option<Vector>> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        // Try binary index first (faster)
        if let Some(vector) = collection.binary_index.get_vector(id)? {
            return Ok(Some(vector));
        }

        // Fallback to SQLite
        collection.sqlite_storage.get_vector(id)
    }

    pub fn delete_vector(&mut self, collection_name: &str, id: &Uuid) -> Result<()> {
        let collection = self.get_collection(collection_name)?
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        // Delete from both storages
        collection.sqlite_storage.delete_vector(id)?;
        collection.binary_index.delete_vector(id)?;

        // Update system info
        let count = collection.binary_index.count_vectors();
        self.set_system_info(collection_name, "vector_count", &count.to_string())?;
        self.set_system_info(collection_name, "updated_at", &get_current_timestamp())?;

        Ok(())
    }

    pub fn get_all_vectors(&self, collection_name: &str) -> Result<Vec<Vector>> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        // Use binary index for better performance
        collection.binary_index.get_all_vectors()
    }

    pub fn count_vectors(&self, collection_name: &str) -> Result<usize> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        Ok(collection.binary_index.count_vectors())
    }

    pub fn get_collection_info(&self, collection_name: &str) -> Result<HashMap<String, String>> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        let mut info = HashMap::new();
        info.insert("name".to_string(), collection.name.clone());
        info.insert("dimension".to_string(), collection.dimension.to_string());
        info.insert("vector_count".to_string(), collection.binary_index.count_vectors().to_string());

        // Get system info from SQLite
        for key in ["created_at", "updated_at", "dimension"] {
            if let Some(value) = collection.sqlite_storage.get_system_info(key)? {
                info.insert(key.to_string(), value);
            }
        }

        Ok(info)
    }

    pub fn set_system_info(&self, collection_name: &str, key: &str, value: &str) -> Result<()> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        collection.sqlite_storage.set_system_info(key, value)
    }

    pub fn optimize_collection(&mut self, collection_name: &str) -> Result<()> {
        let collection = self.get_collection(collection_name)?
            .ok_or_else(|| crate::VectorDBError::StorageError(format!("Collection '{}' not found", collection_name)))?;

        collection.binary_index.optimize()?;

        Ok(())
    }
} 
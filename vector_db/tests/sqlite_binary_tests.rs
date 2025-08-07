use vector_db::{
    vector::Vector,
    collection_manager::CollectionManager,
    sqlite_storage::SQLiteStorage,
    binary_index::BinaryIndex,
    utils::generate_random_vectors,
};
use ndarray::Array1;
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn test_sqlite_storage_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.sqlite3");
    
    let storage = SQLiteStorage::new(&db_path, "test_collection").unwrap();
    
    // Test system info
    storage.set_system_info("test_key", "test_value").unwrap();
    let value = storage.get_system_info("test_key").unwrap();
    assert_eq!(value, Some("test_value".to_string()));
}

#[test]
fn test_sqlite_storage_vector_operations() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.sqlite3");
    
    let storage = SQLiteStorage::new(&db_path, "test_collection").unwrap();
    
    // Create test vector
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let metadata = serde_json::json!({"label": "test"});
    let vector = Vector::with_metadata(data, metadata);
    
    // Test insert
    storage.insert_vector(&vector).unwrap();
    
    // Test get
    let retrieved = storage.get_vector(&vector.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, vector.id);
    assert_eq!(retrieved.data, vector.data);
    assert_eq!(retrieved.metadata, vector.metadata);
    
    // Test count
    assert_eq!(storage.count_vectors().unwrap(), 1);
    
    // Test delete
    storage.delete_vector(&vector.id).unwrap();
    assert_eq!(storage.count_vectors().unwrap(), 0);
    assert!(storage.get_vector(&vector.id).unwrap().is_none());
}

#[test]
fn test_binary_index_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("test.kwi");
    
    let mut index = BinaryIndex::new(&index_path, 3).unwrap();
    
    // Create test vector
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let vector = Vector::new(data);
    
    // Test add
    index.add_vector(&vector).unwrap();
    assert_eq!(index.count_vectors(), 1);
    assert_eq!(index.get_dimension(), 3);
    
    // Test get
    let retrieved = index.get_vector(&vector.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, vector.id);
    assert_eq!(retrieved.data, vector.data);
    
    // Test delete
    index.delete_vector(&vector.id).unwrap();
    assert_eq!(index.count_vectors(), 0);
    assert!(index.get_vector(&vector.id).unwrap().is_none());
}

#[test]
fn test_binary_index_with_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("test.kwi");
    
    let mut index = BinaryIndex::new(&index_path, 3).unwrap();
    
    // Create test vector with metadata
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let metadata = serde_json::json!({"label": "test", "value": 42});
    let vector = Vector::with_metadata(data, metadata);
    
    // Test add
    index.add_vector(&vector).unwrap();
    
    // Test get
    let retrieved = index.get_vector(&vector.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, vector.id);
    assert_eq!(retrieved.data, vector.data);
    assert_eq!(retrieved.metadata, vector.metadata);
}

#[test]
fn test_binary_index_multiple_vectors() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("test.kwi");
    
    let mut index = BinaryIndex::new(&index_path, 128).unwrap();
    
    // Add multiple vectors
    let vectors_data = generate_random_vectors(128, 10);
    let mut vectors = Vec::new();
    
    for data in vectors_data {
        let vector = Vector::new(data);
        vectors.push(vector.clone());
        index.add_vector(&vector).unwrap();
    }
    
    assert_eq!(index.count_vectors(), 10);
    
    // Test get all vectors
    let all_vectors = index.get_all_vectors().unwrap();
    assert_eq!(all_vectors.len(), 10);
    
    // Test individual retrieval
    for vector in &vectors {
        let retrieved = index.get_vector(&vector.id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, vector.id);
        assert_eq!(retrieved.data, vector.data);
    }
}

#[test]
fn test_binary_index_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("test.kwi");
    
    let mut index = BinaryIndex::new(&index_path, 64).unwrap();
    
    // Add vectors in random order
    let vectors_data = generate_random_vectors(64, 5);
    for data in vectors_data {
        let vector = Vector::new(data);
        index.add_vector(&vector).unwrap();
    }
    
    let initial_count = index.count_vectors();
    
    // Optimize
    index.optimize().unwrap();
    
    // Verify optimization didn't lose data
    assert_eq!(index.count_vectors(), initial_count);
    
    // Verify all vectors are still accessible
    let all_vectors = index.get_all_vectors().unwrap();
    assert_eq!(all_vectors.len(), initial_count);
}

#[test]
fn test_collection_manager_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CollectionManager::new(temp_dir.path()).unwrap();
    
    // Create collection
    manager.create_collection("test_collection", 128).unwrap();
    
    // List collections
    let collections = manager.list_collections().unwrap();
    assert!(collections.contains(&"test_collection".to_string()));
    
    // Get collection info
    let info = manager.get_collection_info("test_collection").unwrap();
    assert_eq!(info.get("name").unwrap(), "test_collection");
    assert_eq!(info.get("dimension").unwrap(), "128");
    assert_eq!(info.get("vector_count").unwrap(), "0");
}

#[test]
fn test_collection_manager_vector_operations() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CollectionManager::new(temp_dir.path()).unwrap();
    
    // Create collection
    manager.create_collection("test_collection", 64).unwrap();
    
    // Add vectors
    let vectors_data = generate_random_vectors(64, 5);
    for (i, data) in vectors_data.into_iter().enumerate() {
        let metadata = serde_json::json!({
            "index": i,
            "label": format!("vector_{}", i)
        });
        let vector = Vector::with_metadata(data, metadata);
        manager.add_vector("test_collection", &vector).unwrap();
    }
    
    // Test count
    assert_eq!(manager.count_vectors("test_collection").unwrap(), 5);
    
    // Test get all vectors
    let all_vectors = manager.get_all_vectors("test_collection").unwrap();
    assert_eq!(all_vectors.len(), 5);
    
    // Test individual retrieval
    if let Some(first_vector) = all_vectors.first() {
        let retrieved = manager.get_vector("test_collection", &first_vector.id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, first_vector.id);
        assert_eq!(retrieved.data, first_vector.data);
        assert_eq!(retrieved.metadata, first_vector.metadata);
    }
    
    // Test delete
    if let Some(vector_to_delete) = all_vectors.first() {
        let initial_count = manager.count_vectors("test_collection").unwrap();
        manager.delete_vector("test_collection", &vector_to_delete.id).unwrap();
        let final_count = manager.count_vectors("test_collection").unwrap();
        assert_eq!(final_count, initial_count - 1);
    }
}

#[test]
fn test_collection_manager_dimension_validation() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CollectionManager::new(temp_dir.path()).unwrap();
    
    // Create collection with dimension 64
    manager.create_collection("test_collection", 64).unwrap();
    
    // Try to add vector with wrong dimension
    let wrong_dimension_data = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 3 dimensions
    let vector = Vector::new(wrong_dimension_data);
    
    let result = manager.add_vector("test_collection", &vector);
    assert!(result.is_err());
    
    // Add vector with correct dimension
    let correct_dimension_data = generate_random_vectors(64, 1)[0].clone();
    let vector = Vector::new(correct_dimension_data);
    let result = manager.add_vector("test_collection", &vector);
    assert!(result.is_ok());
}

#[test]
fn test_collection_manager_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("persistence_test");
    
    // Create manager and add data
    {
        let mut manager = CollectionManager::new(&db_path).unwrap();
        manager.create_collection("test_collection", 32).unwrap();
        
        let vectors_data = generate_random_vectors(32, 3);
        for data in vectors_data {
            let vector = Vector::new(data);
            manager.add_vector("test_collection", &vector).unwrap();
        }
    }
    
    // Create new manager and verify data persistence
    {
        let mut manager = CollectionManager::new(&db_path).unwrap();
        
        // Load existing collection
        let collections = manager.list_collections().unwrap();
        assert!(collections.contains(&"test_collection".to_string()));
        
        // Verify data is still there
        let count = manager.count_vectors("test_collection").unwrap();
        assert_eq!(count, 3);
        
        let all_vectors = manager.get_all_vectors("test_collection").unwrap();
        assert_eq!(all_vectors.len(), 3);
    }
}

#[test]
fn test_collection_manager_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CollectionManager::new(temp_dir.path()).unwrap();
    
    // Create collection and add vectors
    manager.create_collection("test_collection", 64).unwrap();
    
    let vectors_data = generate_random_vectors(64, 10);
    for data in vectors_data {
        let vector = Vector::new(data);
        manager.add_vector("test_collection", &vector).unwrap();
    }
    
    // Optimize collection
    manager.optimize_collection("test_collection").unwrap();
    
    // Verify optimization didn't lose data
    let count = manager.count_vectors("test_collection").unwrap();
    assert_eq!(count, 10);
    
    let all_vectors = manager.get_all_vectors("test_collection").unwrap();
    assert_eq!(all_vectors.len(), 10);
} 
use vector_db::{
    vector::Vector,
    local_storage::LocalStorage,
    utils::generate_random_vectors,
};
use ndarray::Array1;
use serde_json::json;
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn test_local_storage_creation() {
    let temp_dir = TempDir::new().unwrap();
    let storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    assert!(storage.get_storage_path().exists());
    assert!(storage.get_storage_path().join("vectors.kwi").exists());
    assert!(storage.get_storage_path().join("metadata.json").exists());
    assert!(storage.get_storage_path().join(".gitignore").exists());
}

#[test]
fn test_local_storage_add_and_retrieve() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Create test vector
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let metadata = json!({"test": "metadata", "value": 42});
    let vector = Vector::with_metadata(data, metadata);
    
    // Add vector
    storage.add_vector(&vector).unwrap();
    
    // Retrieve vector
    let retrieved = storage.get_vector(&vector.id).unwrap();
    assert!(retrieved.is_some());
    
    let retrieved_vector = retrieved.unwrap();
    assert_eq!(retrieved_vector.id, vector.id);
    assert_eq!(retrieved_vector.data, vector.data);
    assert_eq!(retrieved_vector.metadata, vector.metadata);
}

#[test]
fn test_local_storage_multiple_vectors() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Add multiple vectors
    let vectors_data = generate_random_vectors(32, 10);
    let mut added_vectors = Vec::new();
    
    for (i, data) in vectors_data.into_iter().enumerate() {
        let metadata = json!({
            "index": i,
            "label": format!("vector_{}", i)
        });
        let vector = Vector::with_metadata(data, metadata);
        storage.add_vector(&vector).unwrap();
        added_vectors.push(vector);
    }
    
    // Verify count
    let count = storage.get_vector_count().unwrap();
    assert_eq!(count, 10);
    
    // Retrieve all vectors
    let all_vectors = storage.get_all_vectors().unwrap();
    assert_eq!(all_vectors.len(), 10);
    
    // Verify each vector
    for (i, vector) in all_vectors.iter().enumerate() {
        assert_eq!(vector.id, added_vectors[i].id);
        assert_eq!(vector.data, added_vectors[i].data);
        assert_eq!(vector.metadata, added_vectors[i].metadata);
    }
}

#[test]
fn test_local_storage_delete_vector() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Add vectors
    let vectors_data = generate_random_vectors(16, 5);
    let mut vectors = Vec::new();
    
    for data in vectors_data {
        let vector = Vector::new(data);
        storage.add_vector(&vector).unwrap();
        vectors.push(vector);
    }
    
    // Verify initial count
    assert_eq!(storage.get_vector_count().unwrap(), 5);
    
    // Delete a vector
    let vector_to_delete = &vectors[2];
    storage.delete_vector(&vector_to_delete.id).unwrap();
    
    // Verify count decreased
    assert_eq!(storage.get_vector_count().unwrap(), 4);
    
    // Verify vector is gone
    let retrieved = storage.get_vector(&vector_to_delete.id).unwrap();
    assert!(retrieved.is_none());
    
    // Verify other vectors still exist
    for (i, vector) in vectors.iter().enumerate() {
        if i != 2 { // Skip the deleted one
            let retrieved = storage.get_vector(&vector.id).unwrap();
            assert!(retrieved.is_some());
        }
    }
}

#[test]
fn test_local_storage_metadata_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Create vector with complex metadata
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let metadata = json!({
        "user": {
            "id": 12345,
            "name": "test_user",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        },
        "tags": ["test", "vector", "metadata"],
        "created_at": 1234567890
    });
    
    let vector = Vector::with_metadata(data, metadata);
    storage.add_vector(&vector).unwrap();
    
    // Retrieve and verify metadata
    let retrieved = storage.get_vector(&vector.id).unwrap().unwrap();
    assert_eq!(retrieved.metadata, vector.metadata);
    
    // Verify specific metadata fields
    if let Some(metadata) = &retrieved.metadata {
        assert_eq!(metadata["user"]["name"], "test_user");
        assert_eq!(metadata["user"]["id"], 12345);
        assert_eq!(metadata["tags"][0], "test");
    }
}

#[test]
fn test_local_storage_clear() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Add some vectors
    let vectors_data = generate_random_vectors(8, 3);
    for data in vectors_data {
        let vector = Vector::new(data);
        storage.add_vector(&vector).unwrap();
    }
    
    // Verify vectors exist
    assert_eq!(storage.get_vector_count().unwrap(), 3);
    
    // Clear storage
    storage.clear().unwrap();
    
    // Verify storage is empty
    assert_eq!(storage.get_vector_count().unwrap(), 0);
    assert_eq!(storage.get_all_vectors().unwrap().len(), 0);
}

#[test]
fn test_local_storage_info() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    // Get initial info
    let initial_info = storage.get_storage_info().unwrap();
    assert_eq!(initial_info["vector_count"], 0);
    assert_eq!(initial_info["storage_type"], "local_kwi");
    
    // Add a vector
    let vector = Vector::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
    storage.add_vector(&vector).unwrap();
    
    // Get updated info
    let updated_info = storage.get_storage_info().unwrap();
    assert_eq!(updated_info["vector_count"], 1);
    assert!(updated_info["last_updated"].as_u64().is_some());
}

#[test]
fn test_local_storage_persistence_across_instances() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create vector data
    let vector_data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let vector_metadata = json!({"test": "persistence"});
    let vector = Vector::with_metadata(vector_data, vector_metadata);
    
    // Create first storage instance and add vector
    {
        let mut storage = LocalStorage::new(temp_dir.path()).unwrap();
        storage.add_vector(&vector).unwrap();
    }
    
    // Create second storage instance and verify vector exists
    {
        let storage = LocalStorage::new(temp_dir.path()).unwrap();
        let retrieved = storage.get_vector(&vector.id).unwrap();
        assert!(retrieved.is_some());
        
        let retrieved_vector = retrieved.unwrap();
        assert_eq!(retrieved_vector.data, vector.data);
        assert_eq!(retrieved_vector.metadata, vector.metadata);
    }
}

#[test]
fn test_local_storage_gitignore_creation() {
    let temp_dir = TempDir::new().unwrap();
    let storage = LocalStorage::new(temp_dir.path()).unwrap();
    
    let gitignore_path = storage.get_storage_path().join(".gitignore");
    assert!(gitignore_path.exists());
    
    // Verify .gitignore content
    let content = std::fs::read_to_string(&gitignore_path).unwrap();
    assert_eq!(content.trim(), "*");
} 
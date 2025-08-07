use vector_db::{
    vector::Vector,
    storage::InMemoryStorage,
    index::BruteForceIndex,
    query::QueryEngine,
    utils::{generate_random_vectors, cosine_similarity, euclidean_distance},
    persistence::PersistentStorage,
};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_end_to_end_query() {
    // Create test vectors
    let vectors_data = generate_random_vectors(128, 100);
    let mut vectors = Vec::new();
    
    for data in vectors_data {
        vectors.push(Vector::new(data));
    }
    
    // Setup storage and index
    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();
    
    // Insert vectors into storage
    for vector in &vectors {
        storage.insert(vector.clone()).unwrap();
    }
    
    // Build index
    let indexed_data: Vec<_> = storage.all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();
    
    // Create query engine
    let query_engine = QueryEngine::new(&storage, &index);
    
    // Test query
    let query_vector = Vector::new(generate_random_vectors(128, 1)[0].clone());
    let results = query_engine.search(&query_vector, 5).unwrap();
    
    assert_eq!(results.len(), 5);
    assert!(results.len() <= 5);
}

#[test]
fn test_cosine_similarity_ordering() {
    // Create a simple test with known vectors
    let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let v3 = Array1::from_vec(vec![0.0, 0.0, 1.0]);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    
    let vectors = vec![
        Vector::with_id(uuid::Uuid::new_v4(), v1),
        Vector::with_id(uuid::Uuid::new_v4(), v2),
        Vector::with_id(uuid::Uuid::new_v4(), v3),
    ];
    
    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();
    
    for vector in &vectors {
        storage.insert(vector.clone()).unwrap();
    }
    
    let indexed_data: Vec<_> = storage.all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();
    
    let query_engine = QueryEngine::new(&storage, &index);
    let query_vector = Vector::new(query);
    let results = query_engine.search_with_scores(&query_vector, 3).unwrap();
    
    // The first result should have the highest cosine similarity with the query
    assert!(results.len() > 0);
    assert!(results[0].1 >= results[1].1);
    assert!(results[1].1 >= results[2].1);
}

#[test]
fn test_persistence() {
    let vectors_data = generate_random_vectors(64, 10);
    let vectors: Vec<Vector> = vectors_data
        .into_iter()
        .map(Vector::new)
        .collect();
    
    let temp_path = std::env::temp_dir().join("test_vectors.json");
    let persistence = PersistentStorage::new(&temp_path);
    
    // Save vectors
    persistence.save(&vectors).unwrap();
    
    // Load vectors
    let loaded_vectors = persistence.load().unwrap();
    
    assert_eq!(vectors.len(), loaded_vectors.len());
    
    // Verify vectors are the same
    for (original, loaded) in vectors.iter().zip(loaded_vectors.iter()) {
        assert_eq!(original.id, loaded.id);
        assert_eq!(original.data, loaded.data);
    }
    
    // Cleanup
    persistence.clear().unwrap();
}

#[test]
fn test_storage_operations() {
    let mut storage = InMemoryStorage::new();
    
    // Test insert and get
    let vector = Vector::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
    let vector_id = vector.id;
    
    storage.insert(vector).unwrap();
    assert_eq!(storage.count(), 1);
    
    let retrieved = storage.get(&vector_id).unwrap();
    assert_eq!(retrieved.data, Array1::from_vec(vec![1.0, 2.0, 3.0]));
    
    // Test delete
    storage.delete(&vector_id).unwrap();
    assert_eq!(storage.count(), 0);
    assert!(storage.get(&vector_id).is_none());
}

#[test]
fn test_index_operations() {
    let mut index = BruteForceIndex::new();
    
    let vectors_data = generate_random_vectors(32, 5);
    let vectors: Vec<Vector> = vectors_data
        .into_iter()
        .map(Vector::new)
        .collect();
    
    let indexed_data: Vec<_> = vectors
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    
    index.build(&indexed_data).unwrap();
    
    let query = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    let results = index.query(&query, 3).unwrap();
    
    assert_eq!(results.len(), 3);
    
    index.clear();
    let results_after_clear = index.query(&query, 3).unwrap();
    assert_eq!(results_after_clear.len(), 0);
}

#[test]
fn test_distance_metrics() {
    let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let v3 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    
    // Test cosine similarity
    let sim_12 = cosine_similarity(&v1, &v2);
    let sim_13 = cosine_similarity(&v1, &v3);
    
    assert_eq!(sim_12, 0.0); // Perpendicular vectors
    assert_eq!(sim_13, 1.0); // Same direction
    
    // Test euclidean distance
    let dist_12 = euclidean_distance(&v1, &v2);
    let dist_13 = euclidean_distance(&v1, &v3);
    
    assert_eq!(dist_12, 2.0_f32.sqrt());
    assert_eq!(dist_13, 0.0);
}

#[test]
fn test_query_engine_with_metadata() {
    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();
    
    // Create vectors with metadata
    let metadata1 = serde_json::json!({"label": "cat", "confidence": 0.95});
    let metadata2 = serde_json::json!({"label": "dog", "confidence": 0.87});
    
    let vector1 = Vector::with_metadata(
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        metadata1
    );
    let vector2 = Vector::with_metadata(
        Array1::from_vec(vec![0.0, 1.0, 0.0]),
        metadata2
    );
    
    storage.insert(vector1.clone()).unwrap();
    storage.insert(vector2.clone()).unwrap();
    
    let indexed_data: Vec<_> = storage.all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();
    
    let query_engine = QueryEngine::new(&storage, &index);
    let query_vector = Vector::new(Array1::from_vec(vec![1.0, 0.0, 0.0]));
    
    let results = query_engine.search_with_scores(&query_vector, 2).unwrap();
    
    assert_eq!(results.len(), 2);
    
    // Check that metadata is preserved
    let first_result = results[0].0;
    assert!(first_result.metadata.is_some());
} 
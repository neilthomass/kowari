use ndarray::Array1;
use vector_db::{
    index::{BruteForceIndex, HNSWIndex, Index, LSHIndex},
    persistence::PersistentStorage,
    query::QueryEngine,
    storage::{InMemoryStorage, Storage},
    utils::{cosine_similarity, euclidean_distance, generate_random_vectors},
    vector::Vector,
    VectorDBError,
};

#[test]
fn test_end_to_end_query() {
    // Setup storage and index
    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();

    // Insert two deterministic vectors
    let target = Vector::new(Array1::from_vec(vec![1.0, 0.0, 0.0]));
    let target_id = target.id;
    let other = Vector::new(Array1::from_vec(vec![0.0, 1.0, 0.0]));

    storage.insert(target.clone()).unwrap();
    storage.insert(other).unwrap();

    // Build index
    let indexed_data: Vec<_> = storage
        .all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();

    // Query using the target vector and verify it is the top result
    let query_engine = QueryEngine::new(&storage, &index);
    let results = query_engine.search_with_scores(&target, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.id, target_id);
    assert!(results[0].1 > results[1].1);
}

#[test]
fn test_cosine_similarity_ordering() {
    // Create a simple test with known vectors
    let vector1 = Vector::with_id(uuid::Uuid::new_v4(), Array1::from_vec(vec![1.0, 0.0, 0.0]));
    let id1 = vector1.id;
    let vector2 = Vector::with_id(uuid::Uuid::new_v4(), Array1::from_vec(vec![0.0, 1.0, 0.0]));
    let vector3 = Vector::with_id(uuid::Uuid::new_v4(), Array1::from_vec(vec![0.0, 0.0, 1.0]));

    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();

    storage.insert(vector1).unwrap();
    storage.insert(vector2).unwrap();
    storage.insert(vector3).unwrap();

    let indexed_data: Vec<_> = storage
        .all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();

    let query_engine = QueryEngine::new(&storage, &index);
    let query_vector = Vector::new(Array1::from_vec(vec![1.0, 0.0, 0.0]));
    let results = query_engine.search_with_scores(&query_vector, 3).unwrap();

    // The first result should be the vector most similar to the query
    assert_eq!(results[0].0.id, id1);
    assert!(results[0].1 >= results[1].1);
    assert!(results[1].1 >= results[2].1);
}

#[test]
fn test_persistence() {
    let vectors_data = generate_random_vectors(64, 10);
    let vectors: Vec<Vector> = vectors_data.into_iter().map(Vector::new).collect();

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
fn test_storage_duplicate_insert_error() {
    let mut storage = InMemoryStorage::new();

    let vector = Vector::new(Array1::from_vec(vec![1.0, 2.0, 3.0]));
    storage.insert(vector.clone()).unwrap();
    let err = storage.insert(vector).unwrap_err();
    assert!(matches!(err, VectorDBError::DuplicateId(_)));
}

#[test]
fn test_storage_delete_missing_error() {
    let mut storage = InMemoryStorage::new();

    let id = uuid::Uuid::new_v4();
    let err = storage.delete(&id).unwrap_err();
    assert!(matches!(err, VectorDBError::MissingId(_)));
}

#[test]
fn test_index_operations() {
    let mut index = BruteForceIndex::new();

    let vectors_data = generate_random_vectors(32, 5);
    let vectors: Vec<Vector> = vectors_data.into_iter().map(Vector::new).collect();
    let first_id = vectors[0].id;
    let query = vectors[0].data.clone();

    let indexed_data: Vec<_> = vectors.iter().map(|v| (&v.id, &v.data)).collect();
    index.build(&indexed_data).unwrap();

    let results = index.query(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, first_id);

    index.clear();
    let results_after_clear = index.query(&query, 3).unwrap();
    assert!(results_after_clear.is_empty());
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

    let vector1 = Vector::with_metadata(Array1::from_vec(vec![1.0, 0.0, 0.0]), metadata1);
    let vector2 = Vector::with_metadata(Array1::from_vec(vec![0.0, 1.0, 0.0]), metadata2);

    storage.insert(vector1).unwrap();
    storage.insert(vector2).unwrap();

    let indexed_data: Vec<_> = storage
        .all_vectors()
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

#[test]
fn test_lsh_index_query() {
    let vectors_data = generate_random_vectors(16, 20);
    let mut storage = InMemoryStorage::new();
    let mut first_vector: Option<Vector> = None;

    for (i, data) in vectors_data.into_iter().enumerate() {
        let vector = Vector::new(data);
        if i == 0 {
            first_vector = Some(vector.clone());
        }
        storage.insert(vector).unwrap();
    }

    let indexed_data: Vec<_> = storage
        .all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();

    let mut index = LSHIndex::new(8);
    index.build(&indexed_data).unwrap();

    let query = first_vector.unwrap();
    let results = index.query(&query.data, 5).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, query.id);
}

#[test]
fn test_hnsw_index_query() {
    let vectors_data = generate_random_vectors(32, 20);
    let mut first_vector: Option<Vector> = None;
    let vectors: Vec<Vector> = vectors_data
        .into_iter()
        .enumerate()
        .map(|(i, data)| {
            let v = Vector::new(data);
            if i == 0 {
                first_vector = Some(v.clone());
            }
            v
        })
        .collect();

    let indexed_data: Vec<_> = vectors.iter().map(|v| (&v.id, &v.data)).collect();

    let mut index = HNSWIndex::new(8, 16);
    index.build(&indexed_data).unwrap();

    let query = first_vector.unwrap();
    let results = index.query(&query.data, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, query.id);
}

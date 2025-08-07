use vector_db::{
    vector::Vector,
    storage::InMemoryStorage,
    index::BruteForceIndex,
    query::QueryEngine,
    utils::{generate_random_vectors, cosine_similarity},
};
use ndarray::Array1;

#[test]
fn test_core_functionality() {
    // Test vector creation
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let vector = Vector::new(data);
    assert_eq!(vector.dimension(), 3);
    
    // Test storage
    let mut storage = InMemoryStorage::new();
    storage.insert(vector.clone()).unwrap();
    assert_eq!(storage.count(), 1);
    
    // Test index
    let mut index = BruteForceIndex::new();
    let indexed_data = vec![(&vector.id, &vector.data)];
    index.build(&indexed_data).unwrap();
    
    // Test query engine
    let query_engine = QueryEngine::new(&storage, &index);
    let results = query_engine.search(&vector, 1).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_distance_metrics() {
    let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    
    let cosine = cosine_similarity(&v1, &v2);
    assert_eq!(cosine, 0.0); // Perpendicular vectors
}

#[test]
fn test_random_vectors() {
    let vectors = generate_random_vectors(128, 10);
    assert_eq!(vectors.len(), 10);
    assert_eq!(vectors[0].len(), 128);
} 
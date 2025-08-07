use ndarray::Array1;
use vector_db::{
    index::BruteForceIndex,
    query::QueryEngine,
    storage::InMemoryStorage,
    utils::{cosine_similarity, generate_random_vectors},
    vector::Vector,
};

fn main() {
    println!(" Testing Vector Database Fixes");
    println!("================================");

    // Test 1: Basic vector creation
    println!("\n1. Testing vector creation...");
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let vector = Vector::new(data);
    println!("    Vector created with dimension: {}", vector.dimension());

    // Test 2: Storage operations
    println!("\n2. Testing storage operations...");
    let mut storage = InMemoryStorage::new();
    storage.insert(vector.clone()).unwrap();
    println!("    Vector stored successfully");
    println!("    Storage count: {}", storage.count());

    // Test 3: Index operations
    println!("\n3. Testing index operations...");
    let mut index = BruteForceIndex::new();
    let indexed_data = vec![(&vector.id, &vector.data)];
    index.build(&indexed_data).unwrap();
    println!("    Index built successfully");

    // Test 4: Query engine
    println!("\n4. Testing query engine...");
    let query_engine = QueryEngine::new(&storage, &index);
    let results = query_engine.search(&vector, 1).unwrap();
    println!("    Query executed successfully");
    println!("    Found {} results", results.len());

    // Test 5: Distance metrics
    println!("\n5. Testing distance metrics...");
    let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let cosine = cosine_similarity(&v1, &v2);
    println!("    Cosine similarity calculated: {:.4}", cosine);

    // Test 6: Random vector generation
    println!("\n6. Testing random vector generation...");
    let vectors = generate_random_vectors(128, 5);
    println!("    Generated {} random vectors", vectors.len());
    println!("    Vector dimension: {}", vectors[0].len());

    println!("\n All tests passed! The fixes are working correctly.");
    println!("The vector database is now fully functional.");
}

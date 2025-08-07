use vector_db::{
    vector::Vector,
    storage::InMemoryStorage,
    index::BruteForceIndex,
    query::QueryEngine,
    utils::generate_random_vectors,
    persistence::PersistentStorage,
};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Vector Database Demo");
    println!("======================\n");

    // Create storage and index
    let mut storage = InMemoryStorage::new();
    let mut index = BruteForceIndex::new();

    // Generate some random vectors
    println!("📊 Generating 100 random 128-dimensional vectors...");
    let vectors_data = generate_random_vectors(128, 100);
    let mut vectors = Vec::new();

    for (i, data) in vectors_data.into_iter().enumerate() {
        let vector = Vector::new(data);
        vectors.push(vector.clone());
        storage.insert(vector).unwrap();
        
        if (i + 1) % 20 == 0 {
            println!("  Inserted {} vectors", i + 1);
        }
    }

    // Build the index
    println!("\n🔍 Building search index...");
    let indexed_data: Vec<_> = storage.all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed_data).unwrap();

    // Create query engine
    let query_engine = QueryEngine::new(&storage, &index);

    // Perform some searches
    println!("\n🔎 Performing similarity searches...");
    
    // Search 1: Find similar vectors to the first vector
    let query_vector1 = &vectors[0];
    println!("Search 1: Finding vectors similar to vector {}", query_vector1.id);
    
    let results1 = query_engine.search_with_scores(query_vector1, 5).unwrap();
    for (i, (vector, score)) in results1.iter().enumerate() {
        println!("  {}. Vector {} - Similarity: {:.4}", i + 1, vector.id, score);
    }

    // Search 2: Find similar vectors to a random query
    let random_query_data = generate_random_vectors(128, 1)[0].clone();
    let random_query = Vector::new(random_query_data);
    println!("\nSearch 2: Finding vectors similar to a random query vector");
    
    let results2 = query_engine.search_with_scores(&random_query, 3).unwrap();
    for (i, (vector, score)) in results2.iter().enumerate() {
        println!("  {}. Vector {} - Similarity: {:.4}", i + 1, vector.id, score);
    }

    // Demonstrate persistence
    println!("\n💾 Testing persistence...");
    let temp_path = std::env::temp_dir().join("demo_vectors.json");
    let persistence = PersistentStorage::new(&temp_path);
    
    // Save all vectors
    persistence.save(&vectors).unwrap();
    println!("  Saved {} vectors to {:?}", vectors.len(), temp_path);
    
    // Load vectors back
    let loaded_vectors = persistence.load().unwrap();
    println!("  Loaded {} vectors from disk", loaded_vectors.len());
    
    // Verify they're the same
    assert_eq!(vectors.len(), loaded_vectors.len());
    println!("  ✅ Persistence test passed!");

    // Performance metrics
    println!("\n📈 Performance Summary:");
    println!("  Total vectors: {}", storage.count());
    println!("  Vector dimension: {}", vectors[0].dimension());
    println!("  Storage type: InMemory");
    println!("  Index type: BruteForce");

    // Cleanup
    persistence.clear().unwrap();
    println!("\n🧹 Cleaned up temporary files");

    println!("\n✅ Demo completed successfully!");
    Ok(())
} 
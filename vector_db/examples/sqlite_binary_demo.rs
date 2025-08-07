use ndarray::Array1;
use std::collections::HashMap;
use vector_db::{
    collection_manager::CollectionManager,
    index::{BruteForceIndex, Index},
    query::QueryEngine,
    utils::generate_random_vectors,
    vector::Vector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" SQLite + Binary Index Vector Database Demo");
    println!("=============================================\n");

    // Create collection manager
    let db_path = std::env::temp_dir().join("vector_db_demo");
    let mut manager = CollectionManager::new(&db_path)?;
    println!(" Database location: {:?}", db_path);

    // Create a collection
    let collection_name = "demo_collection";
    let dimension = 128;
    manager.create_collection(collection_name, dimension)?;
    println!(
        " Created collection '{}' with dimension {}",
        collection_name, dimension
    );

    // Generate and add vectors
    println!("\n Adding vectors to collection...");
    let vectors_data = generate_random_vectors(dimension, 100);

    for (i, data) in vectors_data.into_iter().enumerate() {
        let metadata = serde_json::json!({
            "index": i,
            "label": format!("vector_{}", i),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string()
        });

        let vector = Vector::with_metadata(data, metadata);
        manager.add_vector(collection_name, &vector)?;

        if (i + 1) % 20 == 0 {
            println!("  Added {} vectors", i + 1);
        }
    }

    // Get collection info
    println!("\n Collection Information:");
    let info = manager.get_collection_info(collection_name)?;
    for (key, value) in &info {
        println!("  {}: {}", key, value);
    }

    // List all collections
    println!("\n Available Collections:");
    let collections = manager.list_collections()?;
    for collection in &collections {
        println!("  - {}", collection);
    }

    // Test vector retrieval
    println!("\n Testing vector retrieval...");
    let all_vectors = manager.get_all_vectors(collection_name)?;
    println!(
        "  Retrieved {} vectors from binary index",
        all_vectors.len()
    );

    // Test individual vector retrieval
    if let Some(first_vector) = all_vectors.first() {
        let retrieved = manager.get_vector(collection_name, &first_vector.id)?;
        match retrieved {
            Some(vector) => {
                println!("   Successfully retrieved vector {}", vector.id);
                if let Some(metadata) = &vector.metadata {
                    println!("   Metadata: {}", metadata);
                }
            }
            None => println!("   Failed to retrieve vector"),
        }
    }

    // Test similarity search with the collection
    println!("\n Testing similarity search...");
    let mut index = BruteForceIndex::new();

    // Build index from collection vectors
    let indexed_data: Vec<_> = all_vectors.iter().map(|v| (&v.id, &v.data)).collect();
    index.build(&indexed_data)?;

    // Perform search directly with the index
    let query_data = generate_random_vectors(dimension, 1)[0].clone();
    let query_vector = Vector::new(query_data);

    let results = index.query_with_similarity(&query_vector.data, 5, true);
    println!("  Top 5 similar vectors:");
    for (i, (vector_id, score)) in results.iter().enumerate() {
        println!(
            "    {}. Vector {} - Similarity: {:.4}",
            i + 1,
            vector_id,
            score
        );
    }

    // Test system info
    println!("\n System Information:");
    let system_keys = ["created_at", "updated_at", "dimension", "vector_count"];
    for key in &system_keys {
        if let Some(value) = manager
            .get_collection(collection_name)?
            .unwrap()
            .sqlite_storage
            .get_system_info(key)?
        {
            println!("  {}: {}", key, value);
        }
    }

    // Test vector deletion
    println!("\n Testing vector deletion...");
    if let Some(vector_to_delete) = all_vectors.first() {
        let initial_count = manager.count_vectors(collection_name)?;
        manager.delete_vector(collection_name, &vector_to_delete.id)?;
        let final_count = manager.count_vectors(collection_name)?;
        println!(
            "  Deleted vector {}: {} -> {} vectors",
            vector_to_delete.id, initial_count, final_count
        );
    }

    // Test optimization
    println!("\n Testing collection optimization...");
    manager.optimize_collection(collection_name)?;
    println!("   Collection optimized");

    // Final collection info
    println!("\n Final Collection Information:");
    let final_info = manager.get_collection_info(collection_name)?;
    for (key, value) in &final_info {
        println!("  {}: {}", key, value);
    }

    // Cleanup
    println!("\n Cleaning up...");
    manager.delete_collection(collection_name)?;
    if db_path.exists() {
        std::fs::remove_dir_all(&db_path)?;
    }
    println!("   Cleanup completed");

    println!("\n SQLite + Binary Index demo completed successfully!");
    Ok(())
}

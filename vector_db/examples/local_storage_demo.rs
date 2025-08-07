use ndarray::Array1;
use serde_json::json;
use vector_db::{
    index::{BruteForceIndex, Index},
    local_storage::LocalStorage,
    utils::generate_random_vectors,
    vector::Vector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Local Storage Vector Database Demo");
    println!("=====================================");

    // Create local storage in current directory
    let mut storage = LocalStorage::new(".")?;

    println!(" Storage location: {:?}", storage.get_storage_path());
    println!(" Local storage initialized");

    // Generate some test vectors
    println!("\n Generating 50 random 64-dimensional vectors...");
    let vectors_data = generate_random_vectors(64, 50);

    for (i, data) in vectors_data.into_iter().enumerate() {
        let metadata = json!({
            "index": i,
            "label": format!("vector_{}", i),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "features": {
                "category": if i % 3 == 0 { "A" } else if i % 3 == 1 { "B" } else { "C" },
                "priority": i % 5 + 1
            }
        });

        let vector = Vector::with_metadata(data, metadata);
        storage.add_vector(&vector)?;

        if (i + 1) % 10 == 0 {
            println!("  Added {} vectors", i + 1);
        }
    }

    // Get storage information
    println!("\n Storage Information:");
    let info = storage.get_storage_info()?;
    for (key, value) in info.as_object().unwrap() {
        println!("  {}: {}", key, value);
    }

    // Get vector count
    let count = storage.get_vector_count()?;
    println!("  Total vectors: {}", count);

    // Test vector retrieval
    println!("\n Testing vector retrieval...");
    let all_vectors = storage.get_all_vectors()?;
    println!(
        "  Retrieved {} vectors from local storage",
        all_vectors.len()
    );

    // Test individual vector retrieval
    if let Some(first_vector) = all_vectors.first() {
        let retrieved = storage.get_vector(&first_vector.id)?;
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

    // Test similarity search
    println!("\n Testing similarity search...");
    let mut index = BruteForceIndex::new();

    // Build index from storage vectors
    let indexed_data: Vec<_> = all_vectors.iter().map(|v| (&v.id, &v.data)).collect();
    index.build(&indexed_data)?;

    // Perform search
    let query_data = generate_random_vectors(64, 1)[0].clone();
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

    // Test vector deletion
    println!("\n Testing vector deletion...");
    if let Some(vector_to_delete) = all_vectors.first() {
        let initial_count = storage.get_vector_count()?;
        storage.delete_vector(&vector_to_delete.id)?;
        let final_count = storage.get_vector_count()?;
        println!(
            "  Deleted vector {}: {} -> {} vectors",
            vector_to_delete.id, initial_count, final_count
        );
    }

    // Test metadata filtering
    println!("\n Testing metadata filtering...");
    let vectors_with_metadata = storage.get_all_vectors()?;
    let category_a_vectors: Vec<_> = vectors_with_metadata
        .iter()
        .filter(|v| {
            if let Some(metadata) = &v.metadata {
                if let Some(features) = metadata.get("features") {
                    if let Some(category) = features.get("category") {
                        return category.as_str() == Some("A");
                    }
                }
            }
            false
        })
        .collect();
    println!("  Found {} vectors in category A", category_a_vectors.len());

    // Show storage structure
    println!("\n Storage Structure:");
    let storage_path = storage.get_storage_path();
    println!("  Storage directory: {:?}", storage_path);

    if storage_path.exists() {
        for entry in std::fs::read_dir(storage_path)? {
            let entry = entry?;
            let file_type = if entry.file_type()?.is_dir() { "" } else { "" };
            println!("    {} {:?}", file_type, entry.file_name());
        }
    }

    // Test persistence across restarts
    println!("\n Testing persistence...");
    let test_vector = Vector::with_metadata(
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
        json!({"test": "persistence", "value": 42}),
    );
    storage.add_vector(&test_vector)?;

    // Simulate restart by creating new storage instance
    let mut new_storage = LocalStorage::new(".")?;
    let retrieved = new_storage.get_vector(&test_vector.id)?;

    match retrieved {
        Some(vector) => {
            println!("   Persistence test passed - vector retrieved after restart");
            if let Some(metadata) = &vector.metadata {
                println!("   Retrieved metadata: {}", metadata);
            }
        }
        None => println!("   Persistence test failed"),
    }

    // Final storage info
    println!("\n Final Storage Information:");
    let final_info = storage.get_storage_info()?;
    for (key, value) in final_info.as_object().unwrap() {
        println!("  {}: {}", key, value);
    }

    println!("\n Local storage demo completed successfully!");
    println!(" Check the .vector_storage directory to see the .kwi files!");

    Ok(())
}

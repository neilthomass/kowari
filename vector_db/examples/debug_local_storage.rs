use ndarray::Array1;
use serde_json::json;
use std::fs;
use std::io::Read;
use tempfile::TempDir;
use vector_db::{local_storage::LocalStorage, vector::Vector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Debugging Local Storage");
    println!("==========================");

    let temp_dir = TempDir::new()?;
    println!(" Temp directory: {:?}", temp_dir.path());

    // Create storage
    let mut storage = LocalStorage::new(temp_dir.path())?;
    println!(" Storage created");

    // Clear any existing data
    storage.clear()?;
    println!(" Storage cleared");

    // Create a simple vector
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let metadata = json!({"test": "metadata", "value": 42});
    let vector = Vector::with_metadata(data, metadata);

    println!(" Vector created: {}", vector.id);
    println!("   Data: {:?}", vector.data);
    println!("   Metadata: {:?}", vector.metadata);

    // Add vector
    storage.add_vector(&vector)?;
    println!(" Vector added to storage");

    // Check count
    let count = storage.get_vector_count()?;
    println!(" Vector count: {}", count);

    // Try to retrieve vector
    let retrieved = storage.get_vector(&vector.id)?;
    match retrieved {
        Some(retrieved_vector) => {
            println!(" Vector retrieved successfully!");
            println!("   ID: {}", retrieved_vector.id);
            println!("   Data: {:?}", retrieved_vector.data);
            println!("   Metadata: {:?}", retrieved_vector.metadata);
        }
        None => {
            println!(" Failed to retrieve vector");

            // Debug: try to get all vectors
            let all_vectors = storage.get_all_vectors()?;
            println!(" Total vectors in storage: {}", all_vectors.len());

            for (i, v) in all_vectors.iter().enumerate() {
                println!("   Vector {}: {}", i, v.id);
            }
        }
    }

    // Check storage info
    let info = storage.get_storage_info()?;
    println!(" Storage info: {:?}", info);

    Ok(())
}

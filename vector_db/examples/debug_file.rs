use ndarray::Array1;
use serde_json::json;
use std::fs;
use tempfile::TempDir;
use vector_db::{local_storage::LocalStorage, vector::Vector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Debug File Content");
    println!("====================");

    let temp_dir = TempDir::new()?;
    println!(" Temp directory: {:?}", temp_dir.path());

    // Create storage
    let mut storage = LocalStorage::new(temp_dir.path())?;
    println!(" Storage created");

    // Create a simple vector
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let metadata = json!({"test": "metadata", "value": 42});
    let vector = Vector::with_metadata(data, metadata);

    println!(" Vector created: {}", vector.id);

    // Add vector
    storage.add_vector(&vector)?;
    println!(" Vector added to storage");

    // Check file content
    let vectors_file = storage.get_storage_path().join("vectors.kwi");
    if vectors_file.exists() {
        println!(" File exists: {:?}", vectors_file);
        let file_size = fs::metadata(&vectors_file)?.len();
        println!(" File size: {} bytes", file_size);

        // Read first 100 bytes
        let content = fs::read(&vectors_file)?;
        println!(" First 100 bytes: {:?}", &content[..content.len().min(100)]);

        // Show hex dump
        println!(" Hex dump:");
        for (i, chunk) in content.chunks(16).enumerate().take(10) {
            let hex: String = chunk.iter().map(|b| format!("{:02x} ", b)).collect();
            let ascii: String = chunk
                .iter()
                .map(|&b| if b >= 32 && b <= 126 { b as char } else { '.' })
                .collect();
            println!("{:04x}: {} |{}|", i * 16, hex, ascii);
        }
    } else {
        println!(" File does not exist");
    }

    Ok(())
}

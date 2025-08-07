<div align="center">

# Kowari

**Fast, local vector database built in Rust**


*Store, search, and manage your vectors locally*

</div>

---

##  Features

- **Multiple Search Algorithms**
  - Brute-force search for exact results
  - Locality-Sensitive Hashing (LSH) for approximate search
  - Hierarchical Navigable Small World (HNSW) for ultra-fast similarity search

- ** Flexible Storage Backends**
  - In-memory storage for lightning-fast access
  - Custom `.kwi` binary format for efficient disk storage
  - SQLite integration for metadata and system information

- ** High Performance**
  - Written in Rust for maximum speed and safety
  - Optimized vector operations with SIMD support
  - Memory-efficient data structures

- ** Privacy First**
  - 100% local - your data never leaves your machine
  - No cloud dependencies or external services
  - Complete control over your vector data

---


## Setup

```bash
git clone https://github.com/neilthomass/kowari.git
cd kowari
cargo test
```

---

## Example

```rust
use vector_db::{
    index::{HNSWIndex, Index},
    query::QueryEngine,
    storage::InMemoryStorage,
    utils::generate_random_vectors,
    vector::Vector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize storage and index
    let mut storage = InMemoryStorage::new();
    let mut index = HNSWIndex::new(16, 32);

    // Add some vectors
    for data in generate_random_vectors(128, 100) {
        let vector = Vector::new(data);
        storage.insert(vector.clone())?;
    }

    // Build the index
    let indexed: Vec<_> = storage
        .all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed)?;

    // Perform similarity search
    let engine = QueryEngine::new(&storage, &index);
    let query = storage.all_vectors()[0];
    let results = engine.search(query, 5)?;

    println!("Top 5 similar vectors:");
    for (i, v) in results.iter().enumerate() {
        println!("{}. Vector ID: {}", i + 1, v.id);
    }
    
    Ok(())
}
```

## Running tests

```bash
cargo test
```


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Please run `cargo test` to verify changes before sending a PR.

**⭐ Don't forget to star this repository if it helped you! ⭐**



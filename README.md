Kowari stores your vector files locally using the file system and a custom .kwi file format. It allows you to perform efficient cosine similarity searches to find the most relevant vectors in your database.

Designed as a free, local alternative to cloud-based vector databases, Kowari ensures your data remains private. With Kowari’s open-source foundation, you can trust that your vectors are secure and never leave your own environment.

![Alt text](kowari.jpeg)

## Features

- **Multiple index implementations**
  - Brute-force search for exact results
  - Locality-Sensitive Hashing (LSH) using random hyperplanes
  - Hierarchical Navigable Small World (HNSW) graph for fast approximate search
- **Query engine** connecting storage and indexes to perform nearest neighbour
  lookups
- **Storage backends**
  - In-memory storage
  - JSON-based persistent storage
- **Vector utilities** for cosine similarity, Euclidean distance and random
  vector generation
- **Examples and tests** demonstrating usage in the `examples/` and `tests/`
  directories


## Setup

```bash
git clone https://github.com/your-username/kowari.git
cd kowari
cargo test
```

Ensure you have a recent Rust toolchain installed via [rustup](https://rustup.rs/) to compile and run the code.

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
    let mut storage = InMemoryStorage::new();
    let mut index = HNSWIndex::new(16, 32);

    for data in generate_random_vectors(128, 100) {
        let vector = Vector::new(data);
        storage.insert(vector.clone())?;
    }

    let indexed: Vec<_> = storage
        .all_vectors()
        .iter()
        .map(|v| (&v.id, &v.data))
        .collect();
    index.build(&indexed)?;

    let engine = QueryEngine::new(&storage, &index);
    let query = storage.all_vectors()[0];
    let results = engine.search(query, 5)?;

    for v in results {
        println!("{}", v.id);
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

If you find this project useful, consider **starring** the repository to show your support.

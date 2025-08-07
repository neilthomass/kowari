pub mod storage;
pub mod index;
pub mod query;
pub mod vector;
pub mod persistence;
pub mod utils;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorDBError {
    #[error("Storage Error: {0}")]
    StorageError(String),
    #[error("Index Error: {0}")]
    IndexError(String),
    #[error("Persistence Error: {0}")]
    PersistenceError(String),
    #[error("Serialization Error: {0}")]
    SerializationError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, VectorDBError>;

// Re-export main types for convenience
pub use storage::{Storage, InMemoryStorage};
pub use index::{Index, BruteForceIndex};
pub use query::QueryEngine;
pub use vector::Vector;
pub use persistence::PersistentStorage;
pub use utils::{cosine_similarity, euclidean_distance}; 
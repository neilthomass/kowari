pub mod index;
pub mod persistence;
pub mod query;
pub mod storage;
pub mod utils;
pub mod vector;

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
pub use index::{BruteForceIndex, HNSWIndex, Index, LSHIndex};
pub use persistence::PersistentStorage;
pub use query::QueryEngine;
pub use storage::{InMemoryStorage, Storage};
pub use utils::{cosine_similarity, euclidean_distance};
pub use vector::Vector;

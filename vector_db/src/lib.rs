pub mod sqlite_storage;
pub mod binary_index;
pub mod collection_manager;
pub mod local_storage;

// Re-export core types from parent crate
pub use kowari::{
    vector::{self, Vector},
    persistence::{self, PersistentStorage},
    utils::{self, cosine_similarity, euclidean_distance},
    storage::{self, Storage, InMemoryStorage},
    index::{self, Index, BruteForceIndex, LSHIndex, HNSWIndex},
    query::{self, QueryEngine},
    Result,
    VectorDBError,
};

// Re-export vector_db specific types
pub use sqlite_storage::SQLiteStorage;
pub use binary_index::BinaryIndex;
pub use collection_manager::{CollectionManager, Collection};
pub use local_storage::LocalStorage; 
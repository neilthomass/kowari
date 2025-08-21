use crate::{vector::Vector, Result, VectorDBError};
use std::collections::HashMap;
use uuid::Uuid;

pub trait Storage {
    fn insert(&mut self, vector: Vector) -> Result<()>;
    fn get(&self, id: &Uuid) -> Option<&Vector>;
    fn delete(&mut self, id: &Uuid) -> Result<()>;
    fn all_vectors(&self) -> Vec<&Vector>;
    fn count(&self) -> usize;
}

pub struct InMemoryStorage {
    vectors: HashMap<Uuid, Vector>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
    }
}

impl Storage for InMemoryStorage {
    fn insert(&mut self, vector: Vector) -> Result<()> {
        let id = vector.id;
        if self.vectors.insert(id, vector).is_some() {
            return Err(VectorDBError::DuplicateId(id));
        }
        Ok(())
    }

    fn get(&self, id: &Uuid) -> Option<&Vector> {
        self.vectors.get(id)
    }

    fn delete(&mut self, id: &Uuid) -> Result<()> {
        if self.vectors.remove(id).is_none() {
            return Err(VectorDBError::MissingId(*id));
        }
        Ok(())
    }

    fn all_vectors(&self) -> Vec<&Vector> {
        self.vectors.values().collect()
    }

    fn count(&self) -> usize {
        self.vectors.len()
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

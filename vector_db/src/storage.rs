use crate::{vector::Vector, Result};
use uuid::Uuid;
use std::collections::HashMap;

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
        self.vectors.insert(vector.id, vector);
        Ok(())
    }

    fn get(&self, id: &Uuid) -> Option<&Vector> {
        self.vectors.get(id)
    }

    fn delete(&mut self, id: &Uuid) -> Result<()> {
        self.vectors.remove(id);
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
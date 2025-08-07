use crate::{storage::Storage, index::Index, vector::Vector, Result};
use ndarray::Array1;
use uuid::Uuid;

pub struct QueryEngine<'a> {
    storage: &'a dyn Storage,
    index: &'a dyn Index,
}

impl<'a> QueryEngine<'a> {
    pub fn new(storage: &'a dyn Storage, index: &'a dyn Index) -> Self {
        Self { storage, index }
    }

    pub fn search(&self, query_vector: &Vector, top_k: usize) -> Result<Vec<&Vector>> {
        let results = self.index.query(&query_vector.data, top_k)?;
        
        let mut vectors = Vec::new();
        for (id, _similarity) in results {
            if let Some(vector) = self.storage.get(&id) {
                vectors.push(vector);
            }
        }
        
        Ok(vectors)
    }

    pub fn search_with_scores(&self, query_vector: &Vector, top_k: usize) -> Result<Vec<(&Vector, f32)>> {
        let results = self.index.query(&query_vector.data, top_k)?;
        
        let mut vectors_with_scores = Vec::new();
        for (id, similarity) in results {
            if let Some(vector) = self.storage.get(&id) {
                vectors_with_scores.push((vector, similarity));
            }
        }
        
        Ok(vectors_with_scores)
    }

    pub fn search_by_vector(&self, query_data: &Array1<f32>, top_k: usize) -> Result<Vec<&Vector>> {
        let results = self.index.query(query_data, top_k)?;
        
        let mut vectors = Vec::new();
        for (id, _similarity) in results {
            if let Some(vector) = self.storage.get(&id) {
                vectors.push(vector);
            }
        }
        
        Ok(vectors)
    }

    pub fn get_vector(&self, id: &Uuid) -> Option<&Vector> {
        self.storage.get(id)
    }

    pub fn count_vectors(&self) -> usize {
        self.storage.count()
    }
} 
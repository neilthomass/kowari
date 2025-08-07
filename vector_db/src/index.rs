use uuid::Uuid;
use crate::Result;
use ndarray::Array1;
use crate::utils::{cosine_similarity, euclidean_distance};

pub trait Index {
    fn build(&mut self, vectors: &[(&Uuid, &Array1<f32>)]) -> Result<()>;
    fn query(&self, query: &Array1<f32>, top_k: usize) -> Vec<(Uuid, f32)>;
    fn clear(&mut self);
}

pub struct BruteForceIndex {
    indexed_vectors: Vec<(Uuid, Array1<f32>)>,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self {
            indexed_vectors: Vec::new(),
        }
    }

    pub fn query_with_similarity(
        &self,
        query: &Array1<f32>,
        top_k: usize,
        use_cosine: bool,
    ) -> Vec<(Uuid, f32)> {
        let mut results: Vec<(Uuid, f32)> = self
            .indexed_vectors
            .iter()
            .map(|(id, vector)| {
                let similarity = if use_cosine {
                    cosine_similarity(query, vector)
                } else {
                    -euclidean_distance(query, vector) // Negative for similarity ordering
                };
                (*id, similarity)
            })
            .collect();

        // Sort by similarity (descending for cosine, ascending for negative euclidean)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        results.into_iter().take(top_k).collect()
    }
}

impl Index for BruteForceIndex {
    fn build(&mut self, vectors: &[(&Uuid, &Array1<f32>)]) -> Result<()> {
        self.indexed_vectors.clear();
        for (id, vector) in vectors {
            self.indexed_vectors.push((**id, (*vector).clone()));
        }
        Ok(())
    }

    fn query(&self, query: &Array1<f32>, top_k: usize) -> Vec<(Uuid, f32)> {
        self.query_with_similarity(query, top_k, true) // Default to cosine similarity
    }

    fn clear(&mut self) {
        self.indexed_vectors.clear();
    }
}

impl Default for BruteForceIndex {
    fn default() -> Self {
        Self::new()
    }
} 
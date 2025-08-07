use crate::utils::{cosine_similarity, euclidean_distance};
use crate::Result;
use ndarray::Array1;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

pub trait Index {
    fn build(&mut self, vectors: &[(&Uuid, &Array1<f32>)]) -> Result<()>;
    fn query(&self, query: &Array1<f32>, top_k: usize) -> Result<Vec<(Uuid, f32)>>;
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
            // `vector` is a reference to a reference, so dereference before cloning
            self.indexed_vectors.push((**id, (*vector).clone()));
        }
        Ok(())
    }

    fn query(&self, query: &Array1<f32>, top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        Ok(self.query_with_similarity(query, top_k, true)) // Default to cosine similarity
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

/// Locality-Sensitive Hashing (LSH) index using random hyperplane projection.
/// This index approximates nearest neighbour search by hashing vectors into
/// buckets based on the sign of their projection onto a set of random
/// hyperplanes. During querying, only vectors that fall into the same bucket as
/// the query are compared, providing a faster albeit approximate search.
pub struct LSHIndex {
    num_planes: usize,
    hyperplanes: Vec<Array1<f32>>,
    buckets: HashMap<u64, Vec<(Uuid, Array1<f32>)>>,
    all_vectors: Vec<(Uuid, Array1<f32>)>,
}

impl LSHIndex {
    /// Create a new LSH index with the specified number of hyperplanes.
    pub fn new(num_planes: usize) -> Self {
        Self {
            num_planes,
            hyperplanes: Vec::new(),
            buckets: HashMap::new(),
            all_vectors: Vec::new(),
        }
    }

    fn compute_hash(&self, vector: &Array1<f32>) -> u64 {
        let mut hash: u64 = 0;
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            if vector.dot(plane) >= 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }

    fn query_bucket(&self, query: &Array1<f32>, top_k: usize) -> Vec<(Uuid, f32)> {
        let hash = self.compute_hash(query);
        let candidates = self.buckets.get(&hash).cloned().unwrap_or_default();

        let mut results: Vec<(Uuid, f32)> = candidates
            .iter()
            .map(|(id, vector)| (*id, cosine_similarity(query, vector)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(top_k).collect()
    }
}

impl Index for LSHIndex {
    fn build(&mut self, vectors: &[(&Uuid, &Array1<f32>)]) -> Result<()> {
        self.buckets.clear();
        self.hyperplanes.clear();
        self.all_vectors.clear();

        if vectors.is_empty() {
            return Ok(());
        }

        let dim = vectors[0].1.len();
        let mut rng = rand::thread_rng();
        self.hyperplanes = (0..self.num_planes)
            .map(|_| {
                Array1::from(
                    (0..dim)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect::<Vec<f32>>(),
                )
            })
            .collect();

        for (id, vector) in vectors {
            let vec_clone = (*vector).clone();
            let hash = self.compute_hash(&vec_clone);
            self.buckets
                .entry(hash)
                .or_insert_with(Vec::new)
                .push((**id, vec_clone.clone()));
            self.all_vectors.push((**id, vec_clone));
        }

        Ok(())
    }

    fn query(&self, query: &Array1<f32>, top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        let mut results = self.query_bucket(query, top_k);

        if results.len() < top_k {
            // Fall back to brute-force over all vectors to ensure we return enough results
            let mut all_results: Vec<(Uuid, f32)> = self
                .all_vectors
                .iter()
                .map(|(id, vector)| (*id, cosine_similarity(query, vector)))
                .collect();
            all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results = all_results.into_iter().take(top_k).collect();
        }

        Ok(results)
    }

    fn clear(&mut self) {
        self.buckets.clear();
        self.hyperplanes.clear();
        self.all_vectors.clear();
    }
}

impl Default for LSHIndex {
    fn default() -> Self {
        Self::new(16)
    }
}

/// A simple implementation of the Hierarchical Navigable Small World (HNSW)
/// graph for approximate nearest neighbour search. This implementation focuses
/// on clarity over performance and is suitable for small datasets.
pub struct HNSWIndex {
    m: usize,
    ef: usize,
    nodes: Vec<HNSWNode>,
    entry: Option<usize>,
    max_level: usize,
}

struct HNSWNode {
    id: Uuid,
    vector: Array1<f32>,
    level: usize,
    neighbours: Vec<Vec<usize>>, // neighbours per level
}

impl HNSWIndex {
    /// Create a new HNSW index.
    pub fn new(m: usize, ef: usize) -> Self {
        Self {
            m,
            ef,
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
        }
    }

    fn random_level(&self) -> usize {
        let mut level = 0usize;
        let mut rng = rand::thread_rng();
        while rng.gen::<f32>() < 0.5 {
            level += 1;
        }
        level
    }

    fn distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        1.0 - cosine_similarity(a, b)
    }

    fn insert_node(&mut self, id: Uuid, vector: Array1<f32>) {
        let level = self.random_level();
        let idx = self.nodes.len();
        let node = HNSWNode {
            id,
            vector,
            level,
            neighbours: vec![Vec::new(); level + 1],
        };
        if self.entry.is_none() {
            self.entry = Some(idx);
            self.max_level = level;
        }
        self.nodes.push(node);

        // Connect to existing nodes
        for i in 0..idx {
            let max_lvl = usize::min(level, self.nodes[i].level);
            for l in 0..=max_lvl {
                self.nodes[idx].neighbours[l].push(i);
                self.nodes[i].neighbours[l].push(idx);

                // Keep only M closest neighbours per level
                if self.nodes[idx].neighbours[l].len() > self.m {
                    self.prune_neighbours(idx, l);
                }
                if self.nodes[i].neighbours[l].len() > self.m {
                    self.prune_neighbours(i, l);
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(idx);
        }
    }

    fn prune_neighbours(&mut self, node_idx: usize, level: usize) {
        let vector = self.nodes[node_idx].vector.clone();
        let mut neigh = self.nodes[node_idx].neighbours[level]
            .clone()
            .into_iter()
            .map(|n| {
                let d = Self::distance(&vector, &self.nodes[n].vector);
                (n, d)
            })
            .collect::<Vec<_>>();
        neigh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neigh.truncate(self.m);
        self.nodes[node_idx].neighbours[level] = neigh.into_iter().map(|(n, _)| n).collect();
    }

    fn greedy_search(&self, query: &Array1<f32>, start: usize, level: usize) -> usize {
        let mut current = start;
        loop {
            let mut changed = false;
            let mut best_dist = Self::distance(query, &self.nodes[current].vector);
            for &n in &self.nodes[current].neighbours[level] {
                let dist = Self::distance(query, &self.nodes[n].vector);
                if dist < best_dist {
                    best_dist = dist;
                    current = n;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }
}

impl Index for HNSWIndex {
    fn build(&mut self, vectors: &[(&Uuid, &Array1<f32>)]) -> Result<()> {
        self.clear();
        for (id, vector) in vectors {
            self.insert_node(**id, (*vector).clone());
        }
        Ok(())
    }

    fn query(&self, query: &Array1<f32>, top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Greedy search through upper layers
        let mut current = self.entry.unwrap();
        for level in (1..=self.max_level).rev() {
            current = self.greedy_search(query, current, level);
        }

        // Breadth-first search at level 0
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(current);
        visited.insert(current);

        while let Some(idx) = queue.pop_front() {
            for &n in &self.nodes[idx].neighbours[0] {
                if visited.len() >= self.ef {
                    break;
                }
                if visited.insert(n) {
                    queue.push_back(n);
                }
            }
            if visited.len() >= self.ef {
                break;
            }
        }

        let mut results: Vec<(Uuid, f32)> = visited
            .into_iter()
            .map(|i| {
                let node = &self.nodes[i];
                (node.id, cosine_similarity(query, &node.vector))
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        Ok(results)
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.entry = None;
        self.max_level = 0;
    }
}

impl Default for HNSWIndex {
    fn default() -> Self {
        Self::new(16, 32)
    }
}

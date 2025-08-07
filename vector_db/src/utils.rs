use ndarray::Array1;
use std::time::Instant;

pub fn cosine_similarity(v1: &Array1<f32>, v2: &Array1<f32>) -> f32 {
    let dot_product = v1.dot(v2);
    let norm1 = v1.dot(v1).sqrt();
    let norm2 = v2.dot(v2).sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

pub fn euclidean_distance(v1: &Array1<f32>, v2: &Array1<f32>) -> f32 {
    let diff = v1 - v2;
    diff.dot(&diff).sqrt()
}

pub fn manhattan_distance(v1: &Array1<f32>, v2: &Array1<f32>) -> f32 {
    let diff = v1 - v2;
    diff.iter().map(|x| x.abs()).sum()
}

pub fn generate_random_vectors(dim: usize, num: usize) -> Vec<Array1<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..num)
        .map(|_| {
            Array1::from_vec(
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            )
        })
        .collect()
}

pub fn normalize_vector(vector: &Array1<f32>) -> Array1<f32> {
    let magnitude = vector.dot(vector).sqrt();
    if magnitude > 0.0 {
        vector / magnitude
    } else {
        vector.clone()
    }
}

pub fn benchmark_search_performance<F>(search_fn: F, iterations: usize) -> f64 
where
    F: Fn() -> std::time::Duration,
{
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _duration = search_fn();
    }
    
    let total_time = start.elapsed();
    total_time.as_secs_f64() / iterations as f64
}

pub fn calculate_recall_at_k(
    expected_results: &[usize],
    actual_results: &[usize],
    k: usize,
) -> f32 {
    if k == 0 {
        return 0.0;
    }
    
    let k = k.min(expected_results.len()).min(actual_results.len());
    let expected_set: std::collections::HashSet<_> = expected_results[..k].iter().collect();
    let actual_set: std::collections::HashSet<_> = actual_results[..k].iter().collect();
    
    let intersection = expected_set.intersection(&actual_set).count();
    intersection as f32 / k as f32
}

pub fn calculate_precision_at_k(
    expected_results: &[usize],
    actual_results: &[usize],
    k: usize,
) -> f32 {
    if k == 0 {
        return 0.0;
    }
    
    let k = k.min(actual_results.len());
    let expected_set: std::collections::HashSet<_> = expected_results.iter().collect();
    let actual_set: std::collections::HashSet<_> = actual_results[..k].iter().collect();
    
    let intersection = actual_set.intersection(&expected_set).count();
    intersection as f32 / k as f32
} 
use serde::{Serialize, Deserialize};
use ndarray::Array1;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: Uuid,
    pub data: Array1<f32>,
    pub metadata: Option<serde_json::Value>,
}

impl Vector {
    pub fn new(data: Array1<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            data,
            metadata: None,
        }
    }

    pub fn with_metadata(data: Array1<f32>, metadata: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            data,
            metadata: Some(metadata),
        }
    }

    pub fn with_id(id: Uuid, data: Array1<f32>) -> Self {
        Self {
            id,
            data,
            metadata: None,
        }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn magnitude(&self) -> f32 {
        self.data.dot(&self.data).sqrt()
    }
} 
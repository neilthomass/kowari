use crate::{vector::Vector, Result};
use anyhow::Context;
use std::path::Path;
use std::fs::File;
use std::io::{Read, Write};

pub struct PersistentStorage {
    file_path: std::path::PathBuf,
}

impl PersistentStorage {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
        }
    }

    pub fn save_vectors(vectors: &[Vector], path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(vectors)
            .context("Failed to serialize vectors to JSON")?;
        
        let mut file = File::create(path)
            .context("Failed to create file for writing")?;
        
        file.write_all(json.as_bytes())
            .context("Failed to write vectors to file")?;
        
        Ok(())
    }

    pub fn load_vectors(path: &Path) -> Result<Vec<Vector>> {
        let mut file = File::open(path)
            .context("Failed to open file for reading")?;
        
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .context("Failed to read file contents")?;
        
        let vectors: Vec<Vector> = serde_json::from_str(&contents)
            .context("Failed to deserialize vectors from JSON")?;
        
        Ok(vectors)
    }

    pub fn save(&self, vectors: &[Vector]) -> Result<()> {
        Self::save_vectors(vectors, &self.file_path)
    }

    pub fn load(&self) -> Result<Vec<Vector>> {
        Self::load_vectors(&self.file_path)
    }

    pub fn append_vector(&self, vector: &Vector) -> Result<()> {
        let mut vectors = if self.file_path.exists() {
            self.load()?
        } else {
            Vec::new()
        };
        
        vectors.push(vector.clone());
        self.save(&vectors)
    }

    pub fn clear(&self) -> Result<()> {
        if self.file_path.exists() {
            std::fs::remove_file(&self.file_path)
                .context("Failed to remove existing file")?;
        }
        Ok(())
    }
} 
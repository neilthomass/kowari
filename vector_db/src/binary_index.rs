use crate::{vector::Vector, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use uuid::Uuid;
use ndarray::Array1;
use std::collections::HashMap;

const KWI_MAGIC: &[u8; 4] = b"KWI\0";
const KWI_VERSION: u32 = 1;

#[derive(Debug)]
pub struct BinaryIndex {
    file_path: std::path::PathBuf,
    dimension: usize,
    vector_count: usize,
    index_entries: HashMap<Uuid, IndexEntry>,
}

#[derive(Debug, Clone)]
struct IndexEntry {
    offset: u64,
    dimension: u32,
    metadata_size: u32,
}

impl BinaryIndex {
    pub fn new<P: AsRef<Path>>(index_path: P, dimension: usize) -> Result<Self> {
        let file_path = index_path.as_ref().to_path_buf();
        
        let mut index = Self {
            file_path,
            dimension,
            vector_count: 0,
            index_entries: HashMap::new(),
        };

        if index.file_path.exists() {
            index.load_index()?;
        } else {
            index.create_new_index()?;
        }

        Ok(index)
    }

    fn create_new_index(&mut self) -> Result<()> {
        let mut file = File::create(&self.file_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create index file: {}", e)))?;

        // Write header
        file.write_all(KWI_MAGIC)?;
        file.write_u32::<LittleEndian>(KWI_VERSION)?;
        file.write_u32::<LittleEndian>(self.dimension as u32)?;
        file.write_u64::<LittleEndian>(0)?; // vector count
        file.write_u64::<LittleEndian>(0)?; // reserved

        Ok(())
    }

    fn load_index(&mut self) -> Result<()> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open index file: {}", e)))?;

        // Read and validate header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != *KWI_MAGIC {
            return Err(crate::VectorDBError::PersistenceError("Invalid KWI file format".to_string()));
        }

        let version = file.read_u32::<LittleEndian>()?;
        if version != KWI_VERSION {
            return Err(crate::VectorDBError::PersistenceError(format!("Unsupported KWI version: {}", version)));
        }

        self.dimension = file.read_u32::<LittleEndian>()? as usize;
        self.vector_count = file.read_u64::<LittleEndian>()? as usize;
        let _reserved = file.read_u64::<LittleEndian>()?;

        // Read index entries
        for _ in 0..self.vector_count {
            let id_bytes = {
                let mut bytes = [0u8; 16];
                file.read_exact(&mut bytes)?;
                bytes
            };
            let id = Uuid::from_bytes(id_bytes);

            let entry = IndexEntry {
                offset: file.read_u64::<LittleEndian>()?,
                dimension: file.read_u32::<LittleEndian>()?,
                metadata_size: file.read_u32::<LittleEndian>()?,
            };

            self.index_entries.insert(id, entry);
        }

        Ok(())
    }

    pub fn add_vector(&mut self, vector: &Vector) -> Result<()> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open index file: {}", e)))?;

        // Seek to end of file
        let file_size = file.metadata()?.len();
        file.seek(SeekFrom::End(0))?;

        let offset = file.stream_position()?;

        // Write vector data
        let data_bytes = bincode::serialize(&vector.data)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize vector data: {}", e)))?;

        let metadata_bytes = if let Some(metadata) = &vector.metadata {
            serde_json::to_vec(metadata)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize metadata: {}", e)))?
        } else {
            Vec::new()
        };

        // Write vector record
        file.write_all(&data_bytes)?;
        file.write_u32::<LittleEndian>(metadata_bytes.len() as u32)?;
        file.write_all(&metadata_bytes)?;

        // Update index entry
        let entry = IndexEntry {
            offset,
            dimension: vector.dimension() as u32,
            metadata_size: metadata_bytes.len() as u32,
        };

        self.index_entries.insert(vector.id, entry);
        self.vector_count += 1;

        // Update header
        self.update_header(&mut file)?;

        Ok(())
    }

    pub fn get_vector(&self, id: &Uuid) -> Result<Option<Vector>> {
        let entry = match self.index_entries.get(id) {
            Some(entry) => entry,
            None => return Ok(None),
        };

        let mut file = File::open(&self.file_path)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open index file: {}", e)))?;

        file.seek(SeekFrom::Start(entry.offset))?;

        // Read vector data
        let data_size = entry.dimension as usize * 4; // f32 = 4 bytes
        let mut data_bytes = vec![0u8; data_size];
        file.read_exact(&mut data_bytes)?;

        let data: Array1<f32> = bincode::deserialize(&data_bytes)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize vector data: {}", e)))?;

        // Read metadata
        let metadata_size = file.read_u32::<LittleEndian>()? as usize;
        let metadata = if metadata_size > 0 {
            let mut metadata_bytes = vec![0u8; metadata_size];
            file.read_exact(&mut metadata_bytes)?;
            Some(serde_json::from_slice(&metadata_bytes)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize metadata: {}", e)))?)
        } else {
            None
        };

        Ok(Some(Vector {
            id: *id,
            data,
            metadata,
        }))
    }

    pub fn get_all_vectors(&self) -> Result<Vec<Vector>> {
        let mut vectors = Vec::new();
        
        for (id, _entry) in &self.index_entries {
            if let Some(vector) = self.get_vector(id)? {
                vectors.push(vector);
            }
        }

        Ok(vectors)
    }

    pub fn delete_vector(&mut self, id: &Uuid) -> Result<()> {
        if self.index_entries.remove(id).is_some() {
            self.vector_count -= 1;
            
            // Update header
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.file_path)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open index file: {}", e)))?;

            self.update_header(&mut file)?;
        }

        Ok(())
    }

    pub fn count_vectors(&self) -> usize {
        self.vector_count
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn update_header(&self, file: &mut File) -> Result<()> {
        file.seek(SeekFrom::Start(16))?; // Skip magic, version, dimension
        file.write_u64::<LittleEndian>(self.vector_count as u64)?;

        // Write updated index entries
        file.seek(SeekFrom::Start(32))?; // Skip header
        
        for (id, entry) in &self.index_entries {
            file.write_all(id.as_bytes())?;
            file.write_u64::<LittleEndian>(entry.offset)?;
            file.write_u32::<LittleEndian>(entry.dimension)?;
            file.write_u32::<LittleEndian>(entry.metadata_size)?;
        }

        Ok(())
    }

    pub fn optimize(&mut self) -> Result<()> {
        // Create a new optimized index file
        let temp_path = self.file_path.with_extension("tmp");
        let mut optimized_index = BinaryIndex::new(&temp_path, self.dimension)?;

        // Re-add all vectors in order
        let vectors = self.get_all_vectors()?;
        for vector in vectors {
            optimized_index.add_vector(&vector)?;
        }

        // Replace old file with optimized one
        std::fs::rename(&temp_path, &self.file_path)?;
        
        // Update self with optimized index
        *self = optimized_index;

        Ok(())
    }
} 
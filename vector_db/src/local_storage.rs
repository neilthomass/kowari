use crate::{vector::Vector, Result};
use std::path::{Path, PathBuf};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use uuid::Uuid;
use ndarray::Array1;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use bincode;
use serde_json;

const KWI_MAGIC: &[u8; 4] = b"KWI\0";
const KWI_VERSION: u32 = 1;
const STORAGE_DIR: &str = ".vector_storage";

#[derive(Debug)]
pub struct LocalStorage {
    base_path: PathBuf,
    storage_dir: PathBuf,
    vectors_file: PathBuf,
    metadata_file: PathBuf,
}

impl LocalStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        let storage_dir = base_path.join(STORAGE_DIR);
        let vectors_file = storage_dir.join("vectors.kwi");
        let metadata_file = storage_dir.join("metadata.json");

        // Create storage directory if it doesn't exist
        fs::create_dir_all(&storage_dir)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create storage directory: {}", e)))?;

        // Create .gitignore to ensure the storage directory is untracked
        let gitignore_path = storage_dir.join(".gitignore");
        if !gitignore_path.exists() {
            let mut gitignore = File::create(&gitignore_path)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create .gitignore: {}", e)))?;
            gitignore.write_all(b"*\n")
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write .gitignore: {}", e)))?;
        }

        let storage = Self {
            base_path,
            storage_dir,
            vectors_file,
            metadata_file,
        };

        // Initialize storage files
        storage.init_storage()?;

        Ok(storage)
    }

    fn init_storage(&self) -> Result<()> {
        // Initialize vectors file if it doesn't exist
        if !self.vectors_file.exists() {
            self.create_vectors_file()?;
        }

        // Initialize metadata file if it doesn't exist
        if !self.metadata_file.exists() {
            self.create_metadata_file()?;
        }

        Ok(())
    }

    fn create_vectors_file(&self) -> Result<()> {
        let mut file = File::create(&self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create vectors file: {}", e)))?;

        // Write header
        file.write_all(KWI_MAGIC)?;
        file.write_u32::<LittleEndian>(KWI_VERSION)?;
        file.write_u64::<LittleEndian>(0)?; // Vector count
        file.write_u32::<LittleEndian>(0)?; // Reserved

        Ok(())
    }

    fn create_metadata_file(&self) -> Result<()> {
        let metadata = serde_json::json!({
            "version": KWI_VERSION,
            "created_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "vector_count": 0,
            "storage_type": "local_kwi"
        });

        let file = File::create(&self.metadata_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create metadata file: {}", e)))?;
        
        serde_json::to_writer_pretty(file, &metadata)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    pub fn add_vector(&mut self, vector: &Vector) -> Result<()> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open vectors file: {}", e)))?;

        // Seek to end of file
        file.seek(SeekFrom::End(0))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek to end: {}", e)))?;

        // Write vector data
        self.write_vector_to_file(&mut file, vector)?;

        // Update header with new count
        self.update_vector_count(&mut file)?;

        // Update metadata
        self.update_metadata()?;

        Ok(())
    }

    fn write_vector_to_file(&self, file: &mut File, vector: &Vector) -> Result<()> {
        // Write vector ID length and string
        let id_str = vector.id.to_string();
        file.write_u32::<LittleEndian>(id_str.len() as u32)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write ID length: {}", e)))?;
        
        let mut id_bytes = [0u8; 36];
        id_str.as_bytes().iter().enumerate().take(36).for_each(|(i, &byte)| id_bytes[i] = byte);
        file.write_all(&id_bytes)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write vector ID: {}", e)))?;

        // Write vector data
        let data_bytes = bincode::serialize(&vector.data)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize vector data: {}", e)))?;
        
        file.write_u32::<LittleEndian>(data_bytes.len() as u32)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write data length: {}", e)))?;
        
        file.write_all(&data_bytes)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write vector data: {}", e)))?;

        // Write metadata if present
        if let Some(metadata) = &vector.metadata {
            let metadata_bytes = serde_json::to_string(metadata)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to serialize metadata: {}", e)))?
                .into_bytes();
            
            file.write_u32::<LittleEndian>(metadata_bytes.len() as u32)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write metadata length: {}", e)))?;
            
            file.write_all(&metadata_bytes)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write metadata: {}", e)))?;
        } else {
            file.write_u32::<LittleEndian>(0)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write zero metadata length: {}", e)))?;
        }

        Ok(())
    }

    fn update_vector_count(&self, file: &mut File) -> Result<()> {
        // Get current count
        file.seek(SeekFrom::Start(8))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek to count position: {}", e)))?;
        
        let current_count = file.read_u64::<LittleEndian>()
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read current count: {}", e)))?;
        
        // Write updated count
        file.seek(SeekFrom::Start(8))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek back to count position: {}", e)))?;
        
        file.write_u64::<LittleEndian>(current_count + 1)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to write updated count: {}", e)))?;

        Ok(())
    }

    fn update_metadata(&self) -> Result<()> {
        let count = self.get_vector_count()?;
        
        let metadata = serde_json::json!({
            "version": KWI_VERSION,
            "created_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "vector_count": count,
            "storage_type": "local_kwi",
            "last_updated": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let file = File::create(&self.metadata_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create metadata file: {}", e)))?;
        
        serde_json::to_writer_pretty(file, &metadata)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    pub fn get_vector(&self, id: &Uuid) -> Result<Option<Vector>> {
        let mut file = File::open(&self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open vectors file: {}", e)))?;

        // Check if file is empty (just header)
        let file_size = file.metadata()?.len();
        if file_size <= 16 {
            return Ok(None); // Empty file
        }

        // Skip header
        file.seek(SeekFrom::Start(16))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek past header: {}", e)))?;

        // Read vectors until we find the one we're looking for
        loop {
            match self.read_vector_from_file(&mut file) {
                Ok(vector) => {
                    if vector.id == *id {
                        return Ok(Some(vector));
                    }
                }
                Err(e) => {
                    if e.to_string().contains("End of file reached") {
                        break; // Normal end of file
                    } else {
                        return Err(e); // Real error
                    }
                }
            }
        }

        Ok(None)
    }

    pub fn get_all_vectors(&self) -> Result<Vec<Vector>> {
        let mut file = File::open(&self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open vectors file: {}", e)))?;

        // Check if file is empty (just header)
        let file_size = file.metadata()?.len();
        if file_size <= 16 {
            return Ok(Vec::new()); // Empty file
        }

        // Skip header
        file.seek(SeekFrom::Start(16))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek past header: {}", e)))?;

        let mut vectors = Vec::new();
        
        loop {
            match self.read_vector_from_file(&mut file) {
                Ok(vector) => vectors.push(vector),
                Err(e) => {
                    if e.to_string().contains("End of file reached") {
                        break; // Normal end of file
                    } else {
                        return Err(e); // Real error
                    }
                }
            }
        }

        Ok(vectors)
    }

    fn read_vector_from_file(&self, file: &mut File) -> Result<Vector> {
        // Read vector ID length and string
        let id_len = match file.read_u32::<LittleEndian>() {
            Ok(len) => len,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    return Err(crate::VectorDBError::PersistenceError("End of file reached".to_string()));
                }
                return Err(crate::VectorDBError::PersistenceError(format!("Failed to read ID length: {}", e)));
            }
        };
        
        println!("DEBUG: Read id_len: {}", id_len);
        
        let mut id_bytes = [0u8; 36];
        match file.read_exact(&mut id_bytes) {
            Ok(_) => {},
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    return Err(crate::VectorDBError::PersistenceError("End of file reached".to_string()));
                }
                return Err(crate::VectorDBError::PersistenceError(format!("Failed to read vector ID: {}", e)));
            }
        }
        
        // Convert bytes to string, trimming null bytes
        let id_str = std::str::from_utf8(&id_bytes)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to parse vector ID: {}", e)))?
            .trim_matches('\0');
        
        println!("DEBUG: id_len: {}, id_str: '{}'", id_len, id_str);
        
        let id = Uuid::parse_str(id_str)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to parse UUID: {}", e)))?;

        // Read data length
        let data_len = file.read_u32::<LittleEndian>()
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read data length: {}", e)))?;

        // Read vector data
        let mut data_bytes = vec![0u8; data_len as usize];
        file.read_exact(&mut data_bytes)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read vector data: {}", e)))?;
        
        let data: Array1<f32> = bincode::deserialize(&data_bytes)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize vector data: {}", e)))?;

        // Read metadata length
        let metadata_len = file.read_u32::<LittleEndian>()
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read metadata length: {}", e)))?;

        // Read metadata if present
        let metadata = if metadata_len > 0 {
            let mut metadata_bytes = vec![0u8; metadata_len as usize];
            file.read_exact(&mut metadata_bytes)
                .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read metadata: {}", e)))?;
            
            let metadata_str = std::str::from_utf8(&metadata_bytes)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to parse metadata string: {}", e)))?;
            
            Some(serde_json::from_str(metadata_str)
                .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to deserialize metadata: {}", e)))?)
        } else {
            None
        };

        Ok(Vector {
            id,
            data,
            metadata,
        })
    }

    pub fn get_vector_count(&self) -> Result<usize> {
        let mut file = File::open(&self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open vectors file: {}", e)))?;

        file.seek(SeekFrom::Start(8))
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to seek to count position: {}", e)))?;
        
        let count = file.read_u64::<LittleEndian>()
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to read count: {}", e)))?;

        Ok(count as usize)
    }

    pub fn delete_vector(&mut self, id: &Uuid) -> Result<()> {
        // For simplicity, we'll rebuild the file without the deleted vector
        let vectors = self.get_all_vectors()?;
        let filtered_vectors: Vec<_> = vectors.into_iter().filter(|v| v.id != *id).collect();

        // Recreate the file
        self.recreate_vectors_file(&filtered_vectors)?;
        
        // Update metadata
        self.update_metadata()?;

        Ok(())
    }

    fn recreate_vectors_file(&self, vectors: &[Vector]) -> Result<()> {
        // Create temporary file
        let temp_file = self.storage_dir.join("vectors_temp.kwi");
        let mut file = File::create(&temp_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to create temp file: {}", e)))?;

        // Write header
        file.write_all(KWI_MAGIC)?;
        file.write_u32::<LittleEndian>(KWI_VERSION)?;
        file.write_u64::<LittleEndian>(vectors.len() as u64)?;
        file.write_u32::<LittleEndian>(0)?; // Reserved

        // Write all vectors
        for vector in vectors {
            self.write_vector_to_file(&mut file, vector)?;
        }

        // Replace original file
        fs::rename(&temp_file, &self.vectors_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to replace vectors file: {}", e)))?;

        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        // Recreate empty vectors file
        self.create_vectors_file()?;
        self.update_metadata()?;

        Ok(())
    }

    pub fn get_storage_info(&self) -> Result<serde_json::Value> {
        let metadata_file = File::open(&self.metadata_file)
            .map_err(|e| crate::VectorDBError::PersistenceError(format!("Failed to open metadata file: {}", e)))?;
        
        let metadata: serde_json::Value = serde_json::from_reader(metadata_file)
            .map_err(|e| crate::VectorDBError::SerializationError(format!("Failed to read metadata: {}", e)))?;

        Ok(metadata)
    }

    pub fn get_storage_path(&self) -> &Path {
        &self.storage_dir
    }
} 
use std::fmt;
use std::error::Error as StdError;
use std::io;

// Módulo para control de errores.


// Define un tipo de error personalizado para el sistema VFS
#[derive(Debug)]
pub enum VFSError {
    IoError(io::Error),
    MemtableError(String),
    InvalidVector(String),
    IdGenerationError(String),
    SerializationError(String),
    // Puedes añadir más variantes según necesites
}

impl fmt::Display for VFSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VFSError::IoError(err) => write!(f, "I/O error: {}", err),
            VFSError::MemtableError(msg) => write!(f, "Memtable error: {}", msg),
            VFSError::InvalidVector(msg) => write!(f, "Invalid vector: {}", msg),
            VFSError::IdGenerationError(msg) => write!(f, "ID generation error: {}", msg),
            VFSError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl StdError for VFSError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            VFSError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

// Implementamos From para convertir fácilmente desde io::Error
impl From<io::Error> for VFSError {
    fn from(err: io::Error) -> Self {
        VFSError::IoError(err)
    }
}

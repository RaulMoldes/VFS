#![feature(portable_simd)]

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::Utc;
use core::simd::Simd;
use std::mem;


// Metadatos del vector
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorMetadata {
    pub name: String,
    pub tags: Vec<String>,
    pub created_at: String,
}

/// Vector usa Vec<f32> internamente, pero puede construirse desde Simd
// TODO: hacerlo dinámico para poder comprimir el vector (cuantizarlo).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Vector {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadata,
}


// Wrapper Enum necesario para trabajar con SIMD.
// Porque el compilador no puede inferir la longitud que va a tener el vector SIMD, así que necesitamos predefinir la longitud nosotros.
// Las longitudes válidas se muestran aquí: https://doc.rust-lang.org/std/simd/trait.SupportedLaneCount.html
pub enum SimdVector {
    Vec4(Simd<f32, 4>),
    Vec8(Simd<f32, 8>),
    Vec16(Simd<f32, 16>),
    Vec32(Simd<f32, 32>),
    Vec64(Simd<f32, 64>),
}

impl Vector {
    // Constructor vector normal.
    pub fn from_vec(vector: Vec<f32>, name: &str, tags: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            vector,
            metadata: VectorMetadata {
                name: name.to_string(),
                tags,
                created_at: Utc::now().to_rfc3339(),
            },
        }
    }


    
    // Crea un vector desde un SimdVector
    pub fn from_simd(simd_vector: SimdVector, name: &str, tags: Vec<String>) -> Self {
        match simd_vector {
            SimdVector::Vec4(vector) => {
                let vector = vector.to_array().to_vec();
                Self {
                    id: Uuid::new_v4(),
                    vector,
                    metadata: VectorMetadata {
                        name: name.to_string(),
                        tags,
                        created_at: Utc::now().to_rfc3339(),
                    },
                }
            }
            SimdVector::Vec8(vector) => {
                let vector = vector.to_array().to_vec();
                Self {
                    id: Uuid::new_v4(),
                    vector,
                    metadata: VectorMetadata {
                        name: name.to_string(),
                        tags,
                        created_at: Utc::now().to_rfc3339(),
                    },
                }
            }
            SimdVector::Vec16(vector) => {
                let vector = vector.to_array().to_vec();
                Self {
                    id: Uuid::new_v4(),
                    vector,
                    metadata: VectorMetadata {
                        name: name.to_string(),
                        tags,
                        created_at: Utc::now().to_rfc3339(),
                    },
                }
            }
            SimdVector::Vec32(vector) => {
                let vector = vector.to_array().to_vec();
                Self {
                    id: Uuid::new_v4(),
                    vector,
                    metadata: VectorMetadata {
                        name: name.to_string(),
                        tags,
                        created_at: Utc::now().to_rfc3339(),
                    },
                }
            }
            SimdVector::Vec64(vector) => {
                let vector = vector.to_array().to_vec();
                Self {
                    id: Uuid::new_v4(),
                    vector,
                    metadata: VectorMetadata {
                        name: name.to_string(),
                        tags,
                        created_at: Utc::now().to_rfc3339(),
                    },
                }
            }
        }
    }

    // Recuperar el SimdVector desde el Vector
    pub fn as_simd(&self) -> Option<SimdVector> {
        // Intentar construir el Simd en función del tamaño del vector
        match self.vector.len() {
            4 => {
                let mut arr = [0.0f32; 4];
                arr.copy_from_slice(&self.vector);
                Some(SimdVector::Vec4(Simd::from_array(arr)))
            }
            8 => {
                let mut arr = [0.0f32; 8];
                arr.copy_from_slice(&self.vector);
                Some(SimdVector::Vec8(Simd::from_array(arr)))
            }
            16 => {
                let mut arr = [0.0f32; 16];
                arr.copy_from_slice(&self.vector);
                Some(SimdVector::Vec16(Simd::from_array(arr)))
            }
            32 => {
                let mut arr = [0.0f32; 32];
                arr.copy_from_slice(&self.vector);
                Some(SimdVector::Vec32(Simd::from_array(arr)))
            }
            64 => {
                let mut arr = [0.0f32; 64];
                arr.copy_from_slice(&self.vector);
                Some(SimdVector::Vec64(Simd::from_array(arr)))
            }
            _ => None, // Si el tamaño no es válido para ningún tamaño de Simd, retornamos None
        }
    }

}
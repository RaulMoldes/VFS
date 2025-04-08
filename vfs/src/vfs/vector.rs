use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::Utc;
use core::simd::Simd;
use std::simd::{SupportedLaneCount, LaneCount};


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
// Porque e

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


    
    // Convertir vector en Simd
    pub fn as_simd<const N: usize>(&self) -> Option<Simd<f32, N>>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        if self.vector.len() == N {
            let array: [f32; N] = self.vector.clone().try_into().ok()?;
            Some(Simd::from_array(array))
        } else {
            None
        }
    }

    pub fn from_simd<const N: usize>(simd_vector: Simd<f32, N>, name: &str, tags: Vec<String>) -> Self
    where
        LaneCount<N>: SupportedLaneCount, 
        // El compilador no puede inferir la longitud que va a tener el vector SIMD, así que necesitamos predefinir la longitud nosotros.
        // Las longitudes válidas se muestran aquí: https://doc.rust-lang.org/std/simd/trait.SupportedLaneCount.html
        
    {
        Vector {
            id: Uuid::new_v4(),
            vector: simd_vector.to_array().to_vec(),
            metadata: VectorMetadata {
                name: name.to_string(),
                tags,
                created_at: Utc::now().to_rfc3339(),
            },
        }
    }

}
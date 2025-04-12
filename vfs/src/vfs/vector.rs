use serde::{Serialize, Deserialize};
//use uuid::Uuid;
use core::simd::Simd;
use std::simd::{SupportedLaneCount, LaneCount};
use std::convert::TryInto;
use chrono::{DateTime, Utc};

// Metadatos del vector
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorMetadata {
    pub manager_name: String,
    pub name: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Vector usa Vec<f32> internamente, pero puede construirse desde Simd
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Vector {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadata,
}

// Struct vector cuantizado. Usa Vec<i8> internamente y solo puede construirse desde un vector no cuantizado.
// Permite descuantizarse, devolviendo un `Vector`
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuantizedVector {
    pub id: u64,
    pub vector: Vec<i8>,
    pub scale_factor: f32,
    pub metadata: VectorMetadata,
}


// Esta es la principal abstracción de VFS-Vector. Es la que debe usarse en otros módulos.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum VFSVector {
    Dense(Vector),
    Quantized(QuantizedVector),
}

impl Vector {
    // Constructor vector normal.
    pub fn from_vec(vector: Vec<f32>, id: u64, name: &str, tags: Vec<String>) -> Self {
        Self {
            id: id,
            vector,
            metadata: VectorMetadata {
                name: name.to_string(),
                tags,
                created_at: Utc::now(),
            },
        }
    }


    
    // Convertir vector en Simd
    pub fn as_simd<const N: usize>(&self) -> Option<Simd<f32, N>>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        if self.vector.len() == N {
            self.vector.as_slice().try_into().ok().map(Simd::from_array)
        } else {
            None
        }
    }

    // Constructor de vector desde Simd.
    pub fn from_simd<const N: usize>(simd_vector: Simd<f32, N>, id: u64, name: &str, tags: Vec<String>) -> Self
    where
        LaneCount<N>: SupportedLaneCount, 
        // El compilador no puede inferir la longitud que va a tener el vector SIMD, así que necesitamos predefinir la longitud nosotros.
        // Las longitudes válidas se muestran aquí: https://doc.rust-lang.org/std/simd/trait.SupportedLaneCount.html
        
    {
        Vector {
            id: id,
            vector: simd_vector.to_array().to_vec(),
            metadata: VectorMetadata {
                name: name.to_string(),
                tags,
                created_at: Utc::now(),
            },
        }
    }

    /// Cuantiza los valores del vector de f32 a i8
    /// Retorna un nuevo Vector con los valores cuantizados
    pub fn quantize(self, scale_factor: Option<f32>) -> QuantizedVector {
        // Factor de escala para mapear el rango de f32 a i8
        // Suponemos que los valores f32 están en el rango [-1.0, 1.0]
        // i8 tiene rango [-128, 127]
        let scale = scale_factor.unwrap_or(127.0);
        
        // Cuantizamos cada valor del vector
        let quantized_values: Vec<i8> = self.vector
        .iter()
        .map(|&val| {
            // Limitamos al rango [-1.0, 1.0] si no se especificó un scale_factor
            let val = if scale_factor.is_none() { val.clamp(-1.0, 1.0) } else { val };
            (val * scale).round() as i8
        })
        .collect();
        
        // Crear un nuevo vector cuantizado
        QuantizedVector {
            id: self.id,
            vector: quantized_values,
            scale_factor: scale,
            metadata: VectorMetadata {
                name: format!("{}_quantized", self.metadata.name),
                tags: {
                    let mut new_tags = self.metadata.tags.clone();
                    new_tags.push("quantized".to_string());
                    new_tags
                },
                created_at: Utc::now(),
            },
        }
    }


}

impl QuantizedVector {

    // Quantized vector no tiene constructor.

    // Método para descomprimir y volver a Vector
    pub fn dequantize(&self) -> Vector {
        let dequantized_values: Vec<f32> = self.vector
            .iter()
            .map(|&val| (val as f32) / self.scale_factor)
            .collect();
        
        Vector {
            id: self.id,
            vector: dequantized_values,
            metadata: VectorMetadata {
                name: self.metadata.name.replace("_quantized", ""),
                tags: {
                    let mut new_tags = self.metadata.tags.clone();
                    new_tags.retain(|tag| tag != "quantized");
                    new_tags
                },
                created_at: Utc::now(),
            },
        }
    }

}




impl VFSVector {
    // Método para obtener ID (común a ambos tipos)
    pub fn id(&self) -> u64 {
        match self {
           VFSVector::Dense(v) => v.id,
           VFSVector::Quantized(v) => v.id,
        }
    }

    // Método para obtener metadatos
    pub fn metadata(&self) -> &VectorMetadata {
        match self {
           VFSVector::Dense(v) => &v.metadata,
           VFSVector::Quantized(v) => &v.metadata,
        }
    }

    

    // Constructor a partir de un vector de f32
    pub fn from_vec(vector: Vec<f32>, id: u64, name: &str, tags: Vec<String>) -> Self {
        let vector = Vector::from_vec(vector, id, name, tags);
       VFSVector::Dense(vector)
    }
    
    // Constructor a partir de un vector de i8 (cuantizado)
    pub fn from_quantized_vec(vector: Vec<i8>, id: u64, scale_factor: f32, name: &str, tags: Vec<String>) -> Self {
        let mut tags = tags.clone();
        tags.push("quantized".to_string());
        
        let quantized = QuantizedVector {
            id: id,
            vector,
            scale_factor,
            metadata: VectorMetadata {
                name: name.to_string(),
                tags,
                created_at: Utc::now(),
            },
        };
        
       VFSVector::Quantized(quantized)
    }


    // Convertir a Simd independientemente del tipo
    pub fn as_simd<const N: usize>(&self) -> Option<Simd<f32, N>>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        match self {
           VFSVector::Dense(v) => v.as_simd(),
           VFSVector::Quantized(qv) => {
                // Para vectores cuantizados, primero descomprimimos o convertimos directamente
                if qv.vector.len() == N {
                    // Descomprimimos cada valor durante la conversión
                    let dequantized: Vec<f32> = qv.vector.iter()
                        .map(|&val| (val as f32) / qv.scale_factor)
                        .collect();
                    
                    let array: [f32; N] = dequantized.try_into().ok()?;
                    Some(Simd::from_array(array))
                } else {
                    None
                }
            }
        }
    }


    // Crear desde Simd - puede crear denso o cuantizado según el parámetro
    pub fn from_simd<const N: usize>(
        simd_vector: Simd<f32, N>,
        id: u64, 
        name: &str, 
        tags: Vec<String>,
        quantize: bool,
        scale_factor: Option<f32>
    ) -> Self
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let vector = Vector::from_simd(simd_vector, id, name, tags);
        
        if quantize {
            // Si se solicita cuantización, convertimos directamente aVFSVector::Quantized
           VFSVector::Quantized(vector.quantize(scale_factor))
        } else {
           VFSVector::Dense(vector)
        }
    }

    // Método para obtener los valores como Vec<f32> (dequantizando si es necesario)
    pub fn as_f32_vec(&self) -> Vec<f32> {
        match self {
           VFSVector::Dense(v) => v.vector.clone(),
           VFSVector::Quantized(qv) => {
                qv.vector.iter()
                   .map(|&val| (val as f32) / qv.scale_factor)
                   .collect()
            }
        }
    }
    
    // Método para obtener los valores como Vec<i8> (cuantizando si es necesario)
    pub fn as_i8_vec(&self, scale_factor: Option<f32>) -> Vec<i8> {
        match self {
           VFSVector::Dense(v) => {
                // Creamos una copia temporal para poder cuantizar
                let temp_vector = v.clone();
                temp_vector.quantize(scale_factor).vector
            },
           VFSVector::Quantized(qv) => qv.vector.clone(),
        }
    }


    
    
}




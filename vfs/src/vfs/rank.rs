
use super::serializer::load_vectors;
use super::vector::VFSVector;

use std::io;
use std::simd::num::SimdFloat;
use super::ann::VFSANNIndex;

// macro para calcular la distancia euclidea simd.
macro_rules! dynamic_simd_euclidean {
    ($vec1:expr, $vec2:expr, [$( $lanes:literal ),*]) => {{
        let len = $vec1.as_f32_vec().len();
        let mut result: Option<f32> = None;

        $(
            if len == $lanes { // $lanes debe estar entre los valores permitidos  de LaneCount en core::Simd
                // Link:  https://doc.rust-lang.org/std/simd/trait.SupportedLaneCount.html
                if let (Some(simd1), Some(simd2)) =
                    ($vec1.as_simd::<$lanes>(), $vec2.as_simd::<$lanes>())
                {
                    let diff = simd1 - simd2; // Restamos los vectores en paralelo
                    result = Some((diff * diff).reduce_sum().sqrt()); // Suma de productos y raíz cuadrada.
                }
            }
        )*

        result.unwrap_or_else(|| panic!("No se puede usar SIMD con longitud {len}"))
    }};
}

// Macro para calcular la distancia coseno simd.
macro_rules! dynamic_simd_cosine {
    ($vec1:expr, $vec2:expr, [$( $lanes:literal ),*]) => {{
        let len = $vec1.as_f32_vec().len();
        let mut result: Option<f32> = None;

        $(
            if len == $lanes {
                if let (Some(simd1), Some(simd2)) =
                    ($vec1.as_simd::<$lanes>(), $vec2.as_simd::<$lanes>())
                {
                    let dot = (simd1 * simd2).reduce_sum(); // Producto vectorial
                    let norm1 = (simd1 * simd1).reduce_sum().sqrt(); // Norma del primer vector
                    let norm2 = (simd2 * simd2).reduce_sum().sqrt(); // Norma del segundo
                    result = Some(1.0 - (dot / (norm1 * norm2))); // Distancia coseno SIMD, calculada como 1 - similitud coseno
                }
            }
        )*

        result.unwrap_or_else(|| panic!("No se puede usar SIMD con longitud {len}"))
    }};
}



/// Define el tipo de búsqueda a realizar.
pub enum SearchType {
    Exact,
    Approximate,
}

// Métodos de cálculo de distancia.
pub enum DistanceMethod {
    Euclidean,
    Cosine,
    SimdEuclidean,
    SimdCosine,
}


/// Estructura que representa una métrica con su tipo de búsqueda asociado.
pub struct Ranker {
    search_type: SearchType,
    distance_method: DistanceMethod
}

impl Ranker {
    /// Constructor para crear una nueva instancia de `Ranker` con el tipo de búsqueda especificado.
    pub fn new(search_type: SearchType, distance_method: DistanceMethod) -> Self {
        Ranker { search_type,  distance_method }
    }

    /// Método para realizar la búsqueda basada en el tipo especificado.
    pub fn search(&self, query: &VFSVector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>) -> io::Result<Vec<(Box<VFSVector>, f32)>> {

        match self.search_type {

            SearchType::Exact => self.exact_search(query,
                path,
                num_vectors_per_iteration, 
                buffer_size,
                result_limit),

            SearchType::Approximate => self.approximate_search(query,
                path,
                num_vectors_per_iteration, 
                buffer_size,
                result_limit),
        }
    }

    
    /// Implementación de la búsqueda exacta.
    /// La búsqueda exacta requiere cargar todos los vectores en memoria en algún momento.
    /// Usamos `load_vectors` para ir cargando los vectores en un buffer en memoria y luego los liberamos a medida que avanzamos.
    /// 
    /// # Parámetros
    /// - `query`: Referencia al vector de consulta.
    /// - `num_vectors_per_iteration`: Número de vectores a cargar por iteración.
    /// - `buffer_size`: Tamaño del buffer opcional para la lectura de vectores.
    /// - `result_limit`: Número de vectores rankeados a devolver.

    /// # Devuelve
    /// Un vector de tuplas que contiene el vector y su distancia con respecto al vector de consulta.
    fn exact_search(&self,
        query: &VFSVector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(Box<VFSVector>, f32)>>  {

        let mut offset = 0;
        let mut results = Vec::new();
        let path = path;
        let limit = result_limit.unwrap_or(5); // Por defecto vale 5.

        // Esta es mi implementación del ordenamiento por lotes para rankear los vectores en función de la distancia
        // Este método se ha usado ampliamente en Bases de Datos Relacionales para implementar el algoritmo ORDER BY, 
        // Así que para este caso de uso no necesito nada más que eso junto con una medida de distancia que me sirva para rankear.
        
        let mut lowest_score = 0;
        loop {
            // Cargar un lote de vectores desde el archivo
            let (vectors, new_offset) = load_vectors(path, offset, num_vectors_per_iteration, buffer_size)?;
            // Si no se cargaron vectores, hemos llegado al final del archivo
            if vectors.is_empty() {
                println!("Final del archivo alcanzado. No hay más vectores a buscar");
                break;
            }

            // Calcular las distancias y almacenar en el buffer de salida:
            for (index, vector) in vectors.iter().enumerate() {
                let distance = self.calculate_distance(query, vector);
                results.push((Box::new(vector.clone()), distance));
            }
            // Ordenar los resultados por distancia en orden ascendente
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if results.len() > limit {
                results.truncate(limit);
                // Actualizar el puntaje más bajo para comparar en futuras iteraciones
                let lowest_score = results.last().unwrap().1;
                // Filtrar resultados que superen el puntaje más bajo
                results.retain(|&(_, score)| score <= lowest_score);
            }


            offset = new_offset;

        }

        // Finalmente, retornar el buffer de salida.
        Ok(results)
    }

    /// Implementación de la búsqueda aproximada.
    fn approximate_search(&self, query: &VFSVector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(Box<VFSVector>, f32)>> {
        // Implementación de la lógica para búsqueda aproximada.
        // Esto podría incluir el uso de estructuras de datos especializadas como árboles de búsqueda o hashes.
        unimplemented!("Búsqueda aproximada aún no implementada.");
    }

    /// Método para calcular la distancia entre dos vectores.
    fn calculate_distance(&self, vector1: &VFSVector, vector2: &VFSVector) -> f32 {
        /// Verificar que los vectores tengan la misma dimensión
        if vector1.as_f32_vec().len() != vector2.as_f32_vec().len() {
            panic!("Los vectores deben tener la misma dimensión");
        }

        
        match self.distance_method {
            // Simple cálculo de distancia euclidiana.
            // Calcular la suma de los cuadrados de las diferencias
            DistanceMethod::Euclidean => {
                vector1.as_f32_vec().iter()
                    .zip(vector2.as_f32_vec().iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
        
            // Calcular el coseno
            DistanceMethod::Cosine => {
                let dot: f32 = vector1.as_f32_vec().iter()
                    .zip(vector2.as_f32_vec().iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let norm_1: f32 = vector1.as_f32_vec().iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_2: f32 = vector2.as_f32_vec().iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

                1.0 - (dot / (norm_1 * norm_2))
            }

            DistanceMethod::SimdEuclidean => {
                dynamic_simd_euclidean!(vector1, vector2, [2, 4, 8, 16, 32, 64])
            }
        
            DistanceMethod::SimdCosine => {
                dynamic_simd_cosine!(vector1, vector2, [2, 4, 8, 16, 32, 64])
            }
    }
    
}}
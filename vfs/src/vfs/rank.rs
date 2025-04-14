
use super::serializer::load_vectors;
use super::vector::VFSVector;
use super::storage_manager::VFSManager;

use std::io;
use std::simd::num::SimdFloat;
use super::ann::VFSANNIndex;
use std::collections::HashMap;
use rand::rngs::SmallRng;


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
    distance_method: DistanceMethod,
    manager: VFSManager
}

impl Ranker {
    /// Constructor para crear una nueva instancia de `Ranker` con el tipo de búsqueda especificado.
    pub fn new(search_type: SearchType, distance_method: DistanceMethod, manager: VFSManager) -> Self {
        Ranker { search_type,  distance_method, manager }
    }

    /// Método para realizar la búsqueda basada en el tipo especificado.
    pub fn search(&mut self, query: &VFSVector,
        num_vectors_per_iteration: usize, 
        result_limit: Option<usize>) -> io::Result<Vec<(u64, f32)>> {

        match self.search_type {

            SearchType::Exact => self.exact_search(query,
                num_vectors_per_iteration, 
                result_limit),

            SearchType::Approximate => self.approximate_search(query,      
                num_vectors_per_iteration,    
                result_limit),
        }
    }

    
    /// Implementación de la búsqueda exacta por lotes
    /// La búsqueda exacta requiere cargar todos los vectores en memoria en algún momento.
    /// Usamos `load_batch` para ir cargando los vectores en un buffer en memoria y luego los liberamos a medida que avanzamos.
    /// 
    /// # Parámetros
    /// - `query`: Referencia al vector de consulta.
    /// - `num_vectors_per_iteration`: Número de vectores a cargar por iteración.
    /// - `result_limit`: Número de vectores rankeados a devolver.

    /// # Devuelve
    /// Un vector de tuplas que contiene el vector y su distancia con respecto al vector de consulta.
    fn exact_search(&mut self,
        query: &VFSVector,
        num_vectors_per_iteration: usize, 
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(u64, f32)>>  {


        let mut results = Vec::new();
        let limit = result_limit.unwrap_or(5); // Por defecto vale 5.

        // Esta es mi implementación del ordenamiento por lotes para rankear los vectores en función de la distancia
        // Este método se ha usado ampliamente en Bases de Datos Relacionales para implementar el algoritmo ORDER BY, 
        // Así que para este caso de uso no necesito nada más que eso junto con una medida de distancia que me sirva para rankear.
        
        let mut lowest_score = 0;
        loop {

            let off = self.manager.get_current_offset();
            // Cargar un lote de vectores desde el archivo
            let vectors = self.manager.load_batch(num_vectors_per_iteration).expect("Error al cargar el batch de vectores");
            // Si no se cargaron vectores, hemos llegado al final del archivo
            if vectors.is_empty() {
                println!("Final del archivo alcanzado. No hay más vectores a buscar");
                break;
            }

            // Calcular las distancias y almacenar en el buffer de salida:
            for (index, vector) in vectors.iter().enumerate() {
                let distance = self.calculate_distance(query, vector);
    
                let id = vector.id();
                results.push((id, distance));
    
                println!("Se obtiene el vector con id: {}", id);
                println!("Se calcula una distancia de: {:.6}", distance);
                println!("------------------------------------");
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



        }

        // Finalmente, retornar el buffer de salida.
        Ok(results)
    }

    /// Implementación de la búsqueda aproximada.
    fn approximate_search(&mut self, query: &VFSVector,
        num_vectors_per_iteration: usize,     
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(u64, f32)>> {
        // Implementación de la lógica para búsqueda aproximada.
        let limit = result_limit.unwrap_or(5);

        // Crear la función de distancia según el método de distancia configurado
        let distance_fn: Box<dyn Fn(&VFSVector, &VFSVector) -> f32> = match self.distance_method {
            DistanceMethod::Euclidean => Box::new(|vfs1: &VFSVector, vfs2: &VFSVector| {
                let v1 = vfs1.as_f32_vec();
                let v2 = vfs2.as_f32_vec();
                v1.iter()
                    .zip(v2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }),
            DistanceMethod::Cosine => Box::new(|vfs1: &VFSVector, vfs2: &VFSVector| {
                let v1 = vfs1.as_f32_vec();
                let v2 = vfs2.as_f32_vec();
                let dot: f32 = v1.iter()
                    .zip(v2.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                let norm_1: f32 = v1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_2: f32 = v2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                
                1.0 - (dot / (norm_1 * norm_2))
            }),
            DistanceMethod::SimdEuclidean => Box::new(|vfs1: &VFSVector, vfs2: &VFSVector| {
                // En este caso, no podemos usar directamente la macro SIMD
                // porque estamos en un closure, así que usamos la versión no-SIMD

                let v1 = vfs1.as_f32_vec();
                let v2 = vfs2.as_f32_vec();
                println!("El cálculo con SIMD no está soportado para la búsqueda aproximada. Usando la distancia euclídea.");
                v1.iter()
                    .zip(v2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }),
            DistanceMethod::SimdCosine => Box::new(|vfs1: &VFSVector, vfs2: &VFSVector| {
                // Aquí también usamos la versión no-SIMD por la misma razón
    
                println!("El cálculo con SIMD no está soportado para la búsqueda aproximada. Usando la distancia coseno.");

                let v1 = vfs1.as_f32_vec();
                let v2 = vfs2.as_f32_vec();

                let dot: f32 = v1.iter()
                    .zip(v2.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                let norm_1: f32 = v1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_2: f32 = v2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                
                1.0 - (dot / (norm_1 * norm_2))
            }),
        };
    

        // Paso 1: Cargar todos los vectores en memoria  por lotes para construir el índice
        
        let ef_construction = 6;
       

        // Variable para almacenar todos los resultados de las búsquedas en diferentes lotes
        let mut all_results: Vec<(u64, f32)> = Vec::new();


        // Paso 2: Construir el índice HNSW
        let mut ann_index = VFSANNIndex::<_, SmallRng>::new(distance_fn, Some(ef_construction));

        loop {
            let vectors = self.manager.load_batch(num_vectors_per_iteration)
            .expect("Error al cargar el batch de vectores");
        
            if vectors.is_empty() {
                println!("Final del archivo alcanzado. No hay más vectores a leer");
             break;
            }
        
            ann_index.insert_many(vectors);
            
             

        
    }

    // Paso 3: Realizar la búsqueda aproximada
    let ann_results = ann_index.query(query)?; // Usar ? para manejar errores

    // Paso 4: Convertir los resultados al formato esperado
    for (vector, distance) in &ann_results {
        let id = vector.id();
        all_results.push((id, *distance));
    }

    if ann_results.len() == 0 as usize {
        return Ok(Vec::new());
    }

    

    // Ordenar todos los resultados por distancia en orden ascendente
    all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limitar a la cantidad de resultados solicitados
    if all_results.len() > limit {
        all_results.truncate(limit);
    }

    println!("Búsqueda aproximada completada: encontrados {} resultados", all_results.len());

    Ok(all_results)
        
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

use super::serializer::load_vectors;
use super::vector::Vector;

use std::io;

/// Define el tipo de búsqueda a realizar.
pub enum SearchType {
    Exact,
    Approximate,
}

/// Estructura que representa una métrica con su tipo de búsqueda asociado.
pub struct Ranker {
    search_type: SearchType,
}

impl Ranker {
    /// Constructor para crear una nueva instancia de `Ranker` con el tipo de búsqueda especificado.
    pub fn new(search_type: SearchType) -> Self {
        Ranker { search_type }
    }

    /// Método para realizar la búsqueda basada en el tipo especificado.
    pub fn search(&self, query: &Vector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>) -> io::Result<Vec<(Box<Vector>, f32)>> {

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
        query: &Vector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(Box<Vector>, f32)>>  {

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
    fn approximate_search(&self, query: &Vector,
        path: &str,
        num_vectors_per_iteration: usize, 
        buffer_size: Option<usize>,
        result_limit: Option<usize>
       
    ) -> io::Result<Vec<(Box<Vector>, f32)>> {
        // Implementación de la lógica para búsqueda aproximada.
        // Esto podría incluir el uso de estructuras de datos especializadas como árboles de búsqueda o hashes.
        unimplemented!("Búsqueda aproximada aún no implementada.");
    }

    /// Método para calcular la distancia entre dos vectores.
    fn calculate_distance(&self, vector1: &Vector, vector2: &Vector) -> f32 {
        /// Verificar que los vectores tengan la misma dimensión
        if vector1.vector.len() != vector2.vector.len() {
            panic!("Los vectores deben tener la misma dimensión");
        }

        // Simple cálculo de distancia euclidiana.
        // Calcular la suma de los cuadrados de las diferencias
        vector1.vector.iter()
            .zip(vector2.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
}
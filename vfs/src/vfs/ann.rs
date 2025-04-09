// Módulo para la realización de búsquedas aproximada.
// La struct `VFSANNIndex` es un wrapper sobre mi implementación de HNSW para poder adaptarla a mi caso de uso actual.
// ----------------------------------------------------------------------------------------------------------
// He realizado algo de investigación sobre el método de búsqueda aproximada que se usa en otras bases de datos vectoriales como Pinecone o Faiss.
// Básicamente, hay tres categorías en cuanto a los algoritmos de ANN (Aproximate Nearest Neighbors):
// - Grafos (HNSW)
// - Árboles
// - Hashes
// El algoritmo HNSW (https://www.pinecone.io/learn/series/faiss/hnsw/) construye un grafo de proximidad multicapa,
// en el que los vértices(nodos) están unidos en base a su proximidad o similitud. La medida de similitud es una medida de distancia, habitualmente
// HNSW se basa en las `Listas de salto probabilístico`, que consisten en una concatenación de varias capas de listas enlazadas.
// Las búsquedas empezaban en la capa más alta y van recorriendo la lista hacia la derecha hasta que encuentran el final,
// entonces se avanza una capa. Los nodos están más interconectados en las capas más bajas
// |--| ------- ---> |--| -----------> |--| C3 
// |--| --> |--| --> |--| -----------> |--| C2
// |--| --> |--| --> |--| --> |--| --> |--| C1
// HNSW hereda esta misma estructura jerárquica. 
//
// La búsqueda vectorial basada en Pequeños Mundos Navegables consiste en un grafo de proximidad con enlaces de diferente longitud.
// Cada nodo del grafo contiene una lista de nodos `amigos` a los que está unido. A partir de ahí, al buscar un elemento en el grafo, se define un 
// punto de entrada y navegar por el mismo hasta alcanzar el camino más corto al objetivo. El algoritmo de búsqueda utilizado está basado
// algoritmo de `Dijkstra` del camino más corto (https://www.geeksforgeeks.org/introduction-to-dijkstras-shortest-path-algorithm/)
// 1. Fase de Zoom-Out: se realiza un pase sobre los vértices de alto grado (aquellos con más links).
// 2. Fase de Zoom-In: se realiza el pase sobre los vértices de menor grado.
// El grado puede definirse como el número de links en un vértice.
//
// La condición de parada es no encontrar vértices más cercanos en la lista de `amigos` del nodo actual.
//
// HNSW es simplemente la evolución natural de NSW unido a las listas jerárquicas con salto probabilístico multi-capa.
// Añadir la H de jerarquía a NSW produce un grafo en el que los links están separados por capas. 
// En la capa más alta los links son más largos, mientras que en las más bajas los links son más cortos y los nodos son más próximos entre sí.
// 
// Durante la búsqueda, entramos por la capa más alta y se empieza por el link más largo. En cada capa, la búsqueda se implementa igual que en NSW.
// Una vez alcanzado el mínimo local, se baja un nivel y se empieza de nuevo por el mismo nodo en un nivel más bajo.
// 
// La construcción del grafo los vectores se insertan de forma iterativa uno por uno. Se empieza en la capa más alta.
// 
//  Dados los parámetros:
// - L: numero de capas.
// - m_L: multiplicador de capa. La probabilidad de que un vector sea insertado en una capa determinada se normaliza con este multiplicador, donde m_L = 0 significa que los vectores se insertan en la capa 0.
// - P[L] = f(L, m_L): función de probabilidad que define la probabilidad de ser insertado en una capa determinada.
// - ef: número de vecinos del vértice insertado.
// - q: nodo o vértice actual.
// - efConstruction: parámetro configurable que incrementa ef en cada iteración.
// - M: número de vecinos a añadir en la fase dos de la construcción.
// - M_max: número máximo de links que puede tener un nodo.
// - M_max0: número máximo de links en la capa 0.

// Se realiza la construcción en dos fases:
// Fase 1: Se empieza en la capa más alta, y después de entrar en el grafo, el algoritmo navega alrededor de los ejes buscando los ef vecinos mas cercanos de q.
// Si se llega a un mínimo local, se avanza una capa y se repite el proceso hasta alcanzar la capa target.

// Fase 2: Se incrementa ef según efConstruction. y se añaden M vecinos en la capa actual, los cuales sirven como puntos de entrada para la capa siguiente.
// La selección de los M vecinos normalmente se basa en coger los vectores más cercanos a q.

// Nota: los creadores de HNSW recomiendan minimizar el overlap entre nodos sobre una misma capa. 
// Esto se logra con valores más pequeños de m_L ya que se empuja a más vectores a la capa 0, auqnue a la vez incrementa el número de trasversals en el grafo.
// El valor óptimo puede estar alrededor de 1/ln(M).


use std::collections::HashMap;
use rand::Rng;
use uuid::Uuid;
use super::vector::VFSVector;

// Nodo de HNSW que contiene un vector y sus vecinos
#[derive(Debug, Clone)]
pub struct Node {
    pub vector: Vec<f32>,
    pub neighbors: Vec<usize>, // Índices de los vecinos en la capa
}


// Implementación de HNSW para f32. En este repositorio hay una implementación dinámica para todos los niveles de precisión,
// si estás interesado: https://github.com/rust-cv/hnsw

pub struct HNSW {
    pub layers: Vec<Vec<Node>>,             // Capas de nodos
    pub M: usize,                           // Número máximo de conexiones en un nodo
    pub ef_construction: usize,             // Número de candidatos en la construcción
    pub ef_search: usize,                   // Número de candidatos en la búsqueda
    pub max_elements: usize,                // Número máximo de elementos
    pub dim: usize,                         // Dimensionalidad de los vectores
    pub distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>, // Función de distancia
}

impl HNSW {
    // Constructor para el índice HNSW con la función de distancia como parámetro
    pub fn new(dim: usize, max_elements: usize, M: usize, ef_construction: usize, distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>) -> Self {
        HNSW {
            layers: vec![],
            M,
            ef_construction,
            ef_search: ef_construction, // Puedes ajustar ef_search a tu gusto
            max_elements,
            dim,
            distance_fn,
        }
    }
    
    // Insertar un nodo en el índice
    pub fn insert(&mut self, id: Uuid, vector: Vec<f32>) {
        let node = Node {
            vector,
            neighbors: Vec::new(),
        };

        let layers_count = self.generate_layer_count();
        let mut layers = vec![node; layers_count];

        // Insertar el nodo en la capa base (más baja)
        self.layers.push(layers.clone());
        
        // Construir las conexiones en las capas
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            if layer_idx > 0 {
                let previous_layer = &self.layers[layer_idx - 1];
                let closest_neighbors = self.find_closest_neighbors(&previous_layer, &layer.vector);
                layer.neighbors.extend(closest_neighbors);
            }
        }
    }
    
    // Genera una cantidad aleatoria de capas para un nodo
    fn generate_layer_count(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut layer_count = 1;

        while rng.gen::<f32>() < 0.5 {
            layer_count += 1;
        }

        layer_count
    }

    // Encontrar los vecinos más cercanos a un nodo
    fn find_closest_neighbors(&self, layer: &Vec<Node>, query: &Vec<f32>) -> Vec<usize> {
        let mut distances = Vec::new();
        
        for (i, node) in layer.iter().enumerate() {
            let distance = (self.distance_fn)(&node.vector, query);
            distances.push((i, distance));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        distances.into_iter().take(self.M).map(|(i, _)| i).collect()
    }

    // Calcular la distancia entre dos vectores utilizando la función de distancia
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        (self.distance_fn)(v1, v2)
    }

    // Buscar los k vecinos más cercanos para un vector de consulta
    pub fn search(&self, query: &Vec<f32>, k: usize) -> Vec<(Uuid, f32)> {
        let mut candidates = Vec::new();

        // Comenzamos desde la capa más alta
        let top_layer = &self.layers[self.layers.len() - 1];
        
        // Realizar una búsqueda en la capa más alta
        let mut closest_neighbors = self.find_closest_neighbors(top_layer, query);

        // Ahora nos movemos a las capas más bajas para refinar la búsqueda
        for layer_idx in (0..self.layers.len()).rev() {
            let current_layer = &self.layers[layer_idx];
            for neighbor in closest_neighbors {
                let candidate_node = &current_layer[neighbor];
                closest_neighbors.extend(self.find_closest_neighbors(current_layer, &candidate_node.vector));
            }
        }

        // Devuelve los k vecinos más cercanos
        closest_neighbors.into_iter()
            .take(k)
            .map(|id| (id, self.distance(&query, &self.layers[0][id].vector))) // Mapea id a distancia
            .collect()
    }
}


pub struct VFSANNIndex {
    hnsw: HNSW,
}

impl VFSANNIndex {
    pub fn new(dim: usize, 
        max_elements: usize, 
        M: usize, 
        ef_construction: usize, 
        distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>) -> Self {

        let hnsw = Hnsw::new(dim, max_elements, M, ef_construction, distance_fn);
        Self { hnsw }
    }

    /// Insertar un vector en el índice
    pub fn insert(&mut self, vector: &VFSVector) {
        let vec = vector.as_f32_vec();
        
        self.hnsw.insert((vector.id(), vec)); // Obtengo el id del vector para que luego me devuelva el id del vector más cercano
    }

    /// Buscar los k vecinos más cercanos
    pub fn search(&self, query: &VFSVector, k: usize) -> Vec<(Uuid, f32)> {
        let vec = query.as_f32_vec();
        let results = self.hnsw.search(&vec, k);
        results.into_iter().map(|neigh| (neigh.d_id.0, neigh.distance)).collect()
    }
}

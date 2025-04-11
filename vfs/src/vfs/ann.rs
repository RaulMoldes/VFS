// He estado investigando sobre los algoritmos de búsqueda vectorial aproximada.
// La principal fuente de información que he utilizado es la siguiente: https://www.pinecone.io/learn/series/faiss/hnsw/

/*
-----------------------------------------------------------------------------------------------------------------------------
######  HNSW  #####
-----------------------------------------------------------------------------------------------------------------------------
Los algoritmos de búsqueda aproximada de vecinos (ANN) se pueden clasificar en tres categorías: árboles, hashes y grafos. 

HNSW pertenece a la categoría de grafos, específicamente a los grafos de proximidad, donde los nodos (vértices) se conectan según su cercanía, generalmente medida con la distancia euclidiana.

HNSW representa una evolución significativa respecto a los grafos de proximidad básicos, convirtiéndose en un grafo jerárquico navegable de pequeño mundo. Su funcionamiento se basa principalmente en dos técnicas clave:

Listas probabilísticas con salto (skip lists).

Grafos navegables de pequeño mundo (navigable small world graphs).

---------------------------------------------------------------------------------------------------------------------------
######   Lista Skip Probabilística (Probability Skip List)  #######

La lista skip probabilística fue introducida en 1990 por William Pugh. Esta estructura permite realizar búsquedas rápidas como en un arreglo ordenado, pero con la ventaja de usar listas enlazadas, lo que facilita y agiliza la inserción de nuevos elementos (algo complicado en arreglos ordenados).

La idea de las skip lists es construir varios niveles de listas enlazadas.

En el nivel superior, los enlaces saltan muchos nodos intermedios.

A medida que descendemos de nivel, estos "saltos" se reducen, enlazando nodos más cercanos.

Para buscar en una skip list:

Se comienza en el nivel más alto, donde los saltos son más largos.

Se avanza hacia la derecha mientras el valor del nodo actual sea menor al que buscamos.

Si se pasa del valor objetivo, se baja un nivel y se continúa la búsqueda desde el nodo anterior.

Esta estructura es clave para lograr búsquedas rápidas y eficientes en estructuras como HNSW.

ESTRUCTURA:

|-| ----------------------> |-|---------> |-|  NIVEL 3
|-| --------> |-| --------> |-|---------> |-|  NIVEL 2
|-| -> |-| -> |-| -> |-| -> |-| -> |-| -> |-|  NIVEL 1
-------------------------------------------------------------------------------------------------------------------------
###### Grafos de Pequeño Mundo Navegables (Navigable Small World Graphs - NSW) ######

La búsqueda vectorial usando grafos de pequeño mundo navegables fue desarrollada en una serie de trabajos entre 2011 y 2014. La idea principal es que, si se construye un grafo de proximidad que combine enlaces de corto y largo alcance, se puede reducir el tiempo de búsqueda a una complejidad logarítmica o polilogarítmica.

En este tipo de grafo:

* Cada nodo (vértice) se conecta con varios otros nodos, llamados "amigos".

* Cada nodo mantiene una lista de amigos, formando así la estructura del grafo.

-> Para realizar una búsqueda (search):

1. Se comienza desde un punto de entrada predefinido.

2. Este punto está conectado a varios nodos cercanos.

3. Se identifica cuál de estos nodos es el más cercano al vector de consulta, y se avanza hacia él.

4. Este proceso se repite hasta encontrar el nodo más próximo al objetivo, aprovechando los enlaces de diferentes alcances para acelerar la búsqueda.

5. La condición de parada es que no haya nodos vecinos más cercanos en la lista de amigos de un nodo en particular.
----------------------------------------------------------------------------------------------------------------------
###### Grafos Jerárquicos de Pequeño Mundo Navegables (Hierarchical Navigable Small World Graphs - HNSW) ######

HNSW (Hierarchical Navigable Small World) es una evolución natural de los grafos NSW, que incorpora ideas de la estructura jerárquica de múltiples capas inspiradas en las listas skip probabilísticas de Pugh.

- En HNSW, los enlaces del grafo se distribuyen en diferentes niveles jerárquicos:

- En las capas superiores están los enlaces más largos (mayor alcance).

- En las capas inferiores, los enlaces son más cortos (conexiones locales).

1. ###### Construcción del Grafo ######

- La construcción del grafo se hace insertando vectores uno por uno.

- El número de capas jerárquicas se controla con un parámetro L.

- La probabilidad de que un vector se inserte en una capa dada se define mediante una función de probabilidad P(L, M_L), normalizada por un multiplicador de nivel (m_L)
​   Si m_L == 0, los vectores se insertan sólo en la capa 0.
   La regla práctica para optimizar m_L es usar m_L = 1/ln(M), donde M es el número esperado de vecinos por nodo.
   Un m_L bajo reduce el solapamiento entre nodos en cada capa, pero aumenta el número de pasos durante la búsqueda

2. ##### Inserción ######

FASE 1

- La construcción comienza desde la capa superior.

- El algoritmo navega de forma voraz por los bordes del grafo para encontrar el vecino más cercano al nuevo vector (q) con ef=1.

- Una vez encontrado un mínimo local, el algoritmo baja una capa y repite el proceso hasta llegar a la capa de inserción elegida.

FASE 2

- Al llegar a la capa de inserción, el valor ef se incrementa según el factor efConstruction (definido por el usuario).

- Se buscan los M vecinos más cercanos al nuevo vector para formar los enlaces hacia q.


Parámetros Adicionales

- M_max: número máximo de enlaces por nodo.

- M_max0: número máximo de enlaces en la capa 0.


3. ##### BÚSQUEDA #####

El proceso de búsqueda es similar a una combinacion de las listas probabilísticas enlazadas con salto junto a los NSW.

Se empieza por la capa más alta, y se busca el mínimo local. Después se baja una capa y se busca de nuevo el mínimo local hasta llegar a la capa 0.

La condición de parada para la inserción es alcanzar un mínimo local en la capa 0.
*/
// Mi implementación de HNSW sólo funciona para medidas de distancia f32, aquí se muestra una implementación más genérica: https://github.com/rust-cv/hnsw
use std::collections::HashMap;
use rand::Rng;
use uuid::Uuid;
use super::vector::VFSVector;

// Nodo de HNSW que contiene un vector y sus vecinos
#[derive(Debug, Clone)]
pub struct Node {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub neighbors: Vec<Uuid>, // IDs de los vecinos en la capa
}

// Implementación de HNSW para f32
pub struct HNSW {
    // Mapa de capas: cada capa es un HashMap que asocia Uuid con nodos
    pub layers: Vec<HashMap<Uuid, Node>>,
    pub entry_points: Vec<Uuid>,           // Puntos de entrada para cada capa
    pub M: usize,                         // Número máximo de conexiones en un nodo
    pub ef_construction: usize,           // Número de candidatos en la construcción
    pub ef_search: usize,                 // Número de candidatos en la búsqueda
    pub max_elements: usize,              // Número máximo de elementos
    pub dim: usize,                       // Dimensionalidad de los vectores
    pub distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>, // Función de distancia
}

impl HNSW {
    // Constructor para el índice HNSW con la función de distancia como parámetro
    pub fn new(dim: usize, max_elements: usize, M: usize, ef_construction: usize, distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>) -> Self {
        HNSW {
            layers: vec![HashMap::new()], // Inicializar con una capa base vacía
            entry_points: Vec::new(),
            M,
            ef_construction,
            ef_search: ef_construction,
            max_elements,
            dim,
            distance_fn,
        }
    }
    
    // Insertar un nodo en el índice
    pub fn insert(&mut self, id: Uuid, vector: Vec<f32>) {
        // Generar el número de capas para este nodo
        let num_layers = self.generate_layer_count();
        
        // Asegurarnos de que tenemos suficientes capas
        while self.layers.len() < num_layers {
            self.layers.push(HashMap::new());
            self.entry_points.push(id); // Este nodo será el punto de entrada para la nueva capa
        }
        
        // Crear nodo para ser insertado
        let node = Node {
            id,
            vector: vector.clone(),
            neighbors: Vec::new(),
        };
        
        // Si es el primer nodo, simplemente lo añadimos a todas las capas
        if self.layers[0].is_empty() {
            for layer in self.layers.iter_mut() {
                layer.insert(id, node.clone());
            }
            self.entry_points = vec![id; self.layers.len()];
            return;
        }
        
        // Para cada capa, comenzando desde la más alta donde el nodo debería estar
        for layer_idx in (0..num_layers).rev() {
            let mut entry_point = if layer_idx < self.entry_points.len() {
                self.entry_points[layer_idx]
            } else if !self.entry_points.is_empty() {
                *self.entry_points.last().unwrap()
            } else {
                // Si no hay puntos de entrada, tomamos uno cualquiera
                if let Some(id) = self.layers[0].keys().next() {
                    *id
                } else {
                    // Si no hay nodos, este es el primero
                    self.layers[0].insert(id, node.clone());
                    self.entry_points = vec![id];
                    return;
                }
            };
            
            // Buscar los ef_construction vecinos más cercanos en esta capa
            let neighbors = self.search_layer(&vector, entry_point, self.ef_construction, layer_idx);
            
            // Crear un nuevo nodo con estos vecinos
            let mut new_node = node.clone();
            new_node.neighbors = neighbors.into_iter().map(|(id, _)| id).take(self.M).collect();
            
            // Añadir el nodo a la capa
            self.layers[layer_idx].insert(id, new_node);
            
            // Para cada vecino, también establecer este nodo como su vecino
            for neighbor_id in self.layers[layer_idx].get(&id).unwrap().neighbors.clone() {
                if let Some(neighbor) = self.layers[layer_idx].get_mut(&neighbor_id) {
                    neighbor.neighbors.push(id);
                    // Limitar el número de vecinos si excede M
                    if neighbor.neighbors.len() > self.M {
                        // Ordenar vecinos por distancia y mantener solo los M más cercanos
                        let mut distances = Vec::new();
                        for neigh_id in &neighbor.neighbors {
                            if let Some(neigh_node) = self.layers[layer_idx].get(neigh_id) {
                                let distance = (&neigh_node.vector, &neighbor.vector);
                                distances.push((*neigh_id, distance));
                            }
                        }
                        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        neighbor.neighbors = distances.into_iter().map(|(id, _)| id).take(self.M).collect();
                    }
                }
            }
            
            // Actualizar el punto de entrada para la siguiente capa
            entry_point = id;
        }
        
        // Actualizar los puntos de entrada si es necesario
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.contains_key(&id) && i < self.entry_points.len() {
                self.entry_points[i] = id; // aqui da un error porque se usa self como mutable pero ya esta tomado como inmutable.
            }
        }
    }
    
    // Genera una cantidad aleatoria de capas para un nodo
    fn generate_layer_count(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut layer_count = 1;
        let level_probability = 1.0 / (self.M as f32).ln();

        while rng.gen::<f32>() < level_probability && layer_count < self.layers.len() + 1 {
            layer_count += 1;
        }

        layer_count
    }

    // Buscar los ef vecinos más cercanos en una capa específica
    fn search_layer(&self, query: &Vec<f32>, entry_point: Uuid, ef: usize, layer_idx: usize) -> Vec<(Uuid, f32)> {
        let layer = &self.layers[layer_idx];
        
        // Conjunto de candidatos (IDs) que ya hemos visitado
        let mut visited = std::collections::HashSet::new();
        visited.insert(entry_point);
        
        // Cola de prioridad para los candidatos más cercanos que debemos explorar
        let mut candidates = std::collections::BinaryHeap::new();
        
        // Distancia inicial al punto de entrada
        let entry_node = layer.get(&entry_point).unwrap();
        let entry_distance = self.distance(query, &entry_node.vector);
        candidates.push(std::cmp::Reverse((entry_distance, entry_point)));
        
        // Conjunto de resultados hasta ahora (los ef vecinos más cercanos)
        let mut results = vec![(entry_point, entry_distance)];
        
        // Mientras haya candidatos por explorar
        while let Some(std::cmp::Reverse((_, candidate_id))) = candidates.pop() {
            let candidate_node = layer.get(&candidate_id).unwrap();
            let candidate_distance = (self.distance_fn)(query, &candidate_node.vector);
            
            // Si el candidato es más lejano que el ef-ésimo resultado actual, terminamos
            if !results.is_empty() && candidate_distance > results.last().unwrap().1 && results.len() >= ef {
                break;
            }
            
            // Explorar los vecinos del candidato
            for neighbor_id in &candidate_node.neighbors {
                if !visited.contains(neighbor_id) {
                    visited.insert(*neighbor_id);
                    
                    let neighbor_node = layer.get(neighbor_id).unwrap();
                    let distance = self.distance(query, &neighbor_node.vector);
                    
                    // Si tenemos menos de ef resultados o este vecino es más cercano que el más lejano
                    if results.len() < ef || distance < results.last().unwrap().1 {
                        candidates.push(std::cmp::Reverse((distance, *neighbor_id)));
                        results.push((*neighbor_id, distance));
                        
                        // Ordenar resultados por distancia (menor primero)
                        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        
                        // Mantener solo los ef resultados más cercanos
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        
        results
    }

    // Calcular la distancia entre dos vectores utilizando la función de distancia
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        (self.distance_fn)(v1, v2)
    }

    // Buscar los k vecinos más cercanos para un vector de consulta
    pub fn search(&self, query: &Vec<f32>, k: usize) -> Vec<(Uuid, f32)> {
        if self.layers.is_empty() || self.layers[0].is_empty() {
            return Vec::new();
        }
        
        let mut entry_point = self.entry_points[0];
        
        // Descender por las capas, comenzando desde la más alta
        for layer_idx in (1..self.layers.len()).rev() {
            let layer_results = self.search_layer(query, entry_point, 1, layer_idx);
            if !layer_results.is_empty() {
                entry_point = layer_results[0].0;
            }
        }
        
        // Búsqueda final en la capa base (capa 0) con ef_search
        let results = self.search_layer(query, entry_point, self.ef_search, 0);
        
        // Tomar los k vecinos más cercanos
        results.into_iter().take(k).collect()
    }
}

pub struct VFSANNIndex {
    hnsw: HNSW,
    id_to_index: HashMap<Uuid, usize>, // Mapeamos UUIDs a índices internos
    index_counter: usize,              // Contador para asignar nuevos índices
}

impl VFSANNIndex {
    pub fn new(dim: usize, 
        max_elements: usize, 
        M: usize, 
        ef_construction: usize, 
        distance_fn: Box<dyn Fn(&Vec<f32>, &Vec<f32>) -> f32>) -> Self {

        let hnsw = HNSW::new(dim, max_elements, M, ef_construction, distance_fn);
        Self { 
            hnsw,
            id_to_index: HashMap::new(),
            index_counter: 0,
        }
    }

    /// Insertar un vector en el índice
    pub fn insert(&mut self, vector: &VFSVector) {
        let vec = vector.as_f32_vec();
        let id = vector.id();
        
        // Insertar en el índice HNSW
        self.hnsw.insert(id, vec);
        
        // Registrar el mapeo de UUID a índice
        self.id_to_index.insert(id, self.index_counter);
        self.index_counter += 1;
    }

    /// Buscar los k vecinos más cercanos
    pub fn search(&self, query: &VFSVector, k: usize) -> Vec<(Uuid, f32)> {
        let vec = query.as_f32_vec();
        self.hnsw.search(&vec, k)
    }
}
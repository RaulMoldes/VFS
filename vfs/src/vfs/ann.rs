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
//  Parte del código fue adaptado de este repositorio: https://github.com/rust-cv/hnsw
use core::{
    iter::{Cloned, TakeWhile},
    slice::Iter,
};
use std::collections::HashSet;
use super::vector::VFSVector;
use rand_core::{RngCore, SeedableRng};
use uuid::Uuid;
// Enum que representa los dos tipos de capa (Zero y NonZero)
pub enum Layer<T> {
    Zero,
    NonZero(T),
}

// Struct que representa el vecino de un nodo.
#[derive(Copy, Clone, Debug)]
struct Neighbor {
  
    index: usize, // Índice del nodo en el grafo
    distance: f32, // En mi implementación todas las distancias son f32.
}


// Implementación de un buscador de HNSW
#[derive(Clone, Debug)]
struct Searcher {
    candidates: Vec<Neighbor>,
    nearest: Vec<Neighbor>,
    seen: HashSet<usize>, // más eficiente
}

impl Searcher {
    pub fn new(candidates: Vec<Neighbor>) -> Self {
        Self {
            candidates,
            nearest: Vec::new(),
            seen: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}
// Trait para obtener los vecinos de un nodo.
pub trait HasNeighbors<'a, 'b> {
    type NeighborIter: Iterator<Item = usize> + 'a;

    fn get_neighbors(&'b self) -> Self::NeighborIter;
}


// Nodo de capa zero
#[derive(Clone, Debug)]
pub struct NeighborNodes<const N: usize> {
    // Vecinos de este nodo
    pub neighbors: [usize; N],
}

//
impl<'a, 'b: 'a, // b' vive al menos tanto como a'
// N es el número de vecinos.
const N: usize> HasNeighbors<'a, 'b> for NeighborNodes<N> {
    /// Define el tipo de iterador que se usa para iterar sobre los vecinos.
    // Cloned hace que se clonen los valores, de manera que se devuelve usize por valor, no por referencia.
    // Take While toma elementos siempre que una condición se cumple.
    type NeighborIter = TakeWhile<Cloned<Iter<'a, usize>>, fn(&usize) -> bool>;

    fn get_neighbors(&'b self) -> Self::NeighborIter {
        self.neighbors.iter().cloned().take_while(|&n| n != !0) // Para los vecinos de la capa 0, se devuelven los vecinos siempre que n != 0
    }
}




/// Un nodo de cualquier otra capa.
pub struct Node<const N: usize> {
    /// El nodo de la capa cero al que apunta este nodo.
    pub zero_node: usize,
    /// El nodo en la capa siguiente al que este nodo apunta.
    pub next_node: usize,
    /// Los vecinos de este nodo.
    pub neighbors: NeighborNodes<N>,
}

impl<'a, 'b: 'a, const N: usize> HasNeighbors<'a, 'b> for Node<N> {
    type NeighborIter = TakeWhile<Cloned<Iter<'a, usize>>, fn(&usize) -> bool>;

    fn get_neighbors(&'b self) -> Self::NeighborIter {
        self.neighbors.get_neighbors() // Se llama recursivamente a la función get neighbors, hasta llegar a la capa 0.

    }
}




// ---------------------------------------------------------------------------------------------------------------------------------------------------//

// IMPLEMENTACIÓN DE HNSW //
pub struct Hnsw<F, T, R, const M: usize, const M0: usize> 
where
    F: Fn(&T, &T) -> f32,
{
    /// Función de distancia entre dos VFSVector:
    distance_fn: F, // debe ser una función que devuelva float32
    /// Contiene la capa zero.
    zero: Vec<NeighborNodes<M0>>,
    /// Features de la capa zero
    features: Vec<T>,
    /// Cada una de las capas no-zero
    layers: Vec<Vec<Node<M>>>,
    /// Generador pseudoaleatorio
    prng: R,
    /// EfConstruction
    ef_construction: usize,


}

impl<F, T, R, const M: usize, const M0: usize> Hnsw<F, T, R, M, M0>
where
    R: RngCore + SeedableRng,
    F: Fn(&T, &T) -> f32
{
    /// Crea un nuevo índice HNSW
    pub fn new(distance_fn: F, ef_construction: Option<usize>) -> Self {
        Self {
            distance_fn,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            ef_construction: ef_construction.unwrap_or(400) // por defecto le pongo 400
        }
    }


    fn initialize_searcher(&self, q: &T, qid: Uuid, searcher: &mut Searcher) {
        // Clear the searcher.
        searcher.clear();
        // Calculamos la distancia al punto de entrada.
        let entry_distance = (self.distance_fn)(q, self.entry_feature());
        let candidate = Neighbor {
            vid: qid,
            index: 0,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(
            // Marcamos el primer nodo como visto.
            self.layers
                .last()
                .map(|layer| layer[0].zero_node)
                .unwrap_or(0),
        );
    }


    /// Realiza una búsqueda en una sola capa.
    pub fn search_layer<'a>(
        &self,
        q: &T, // Valor buscado
        ef: usize, // ef
        level: usize, // nivel. Si nivel == 0, buscamos recursivamente hasta llegar a la ultima capa.
        searcher: &mut Searcher, // Buscador que guarda el estado de la búsqueda
        dest: &'a mut [Neighbor], // Output
    ) -> &'a mut [Neighbor] {
        // Check por si se pasa un número de capa no-válido.
        if self.features.is_empty() || level >= self.layers() {
            return &mut [];
        }
        // Inicializamos el searcher.
        self.initialize_searcher(q, searcher);
        let cap = 1; // Inicializa el parámetro cap. Vale uno en las capas más altas y se cambia a ef en la capa 0.
        // Cap (capacity) es el tope de elementos cercanos en una capa.

        // Iterar sobre las capas de abajo a arriba
        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_single_layer(q, searcher, Layer::NonZero(layer), cap);
            if ix + 1 == level {
                let found = core::cmp::min(dest.len(), searcher.nearest.len());
                dest.copy_from_slice(&searcher.nearest[..found]);
                return &mut dest[..found];
            }
            self.lower_search(layer, searcher);
        }

        let cap = ef;

        // Buscamos en la última capa
        self.search_zero_layer(q, searcher, cap);

        let found = core::cmp::min(dest.len(), searcher.nearest.len());
        dest.copy_from_slice(&searcher.nearest[..found]);
        &mut dest[..found]
    }



    /// Método que realiza la búsqueda.

    //  - `q` es el elemento a buscar (query)
    //  - `dest` es la capa de destino.
    /// - `ef` es el tamaño del candidate pool. Puede incrementarse para mayor recall, aunque  reduciría la velocidad.
    /// - Si `ef` es menor que `dest.len()`, `dest` se llenará con `ef` elementos.
    ///
    /// Devuelve un fragmento de los vecinos rellenados.
    pub fn nearest<'a>(
        &self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher,
        dest: &'a mut [Neighbor],
    ) -> &'a mut [Neighbor] {
        self.search_layer(q, ef, 0, searcher, dest)
    }

    /// Encuentra los vecinos más cercanos a `q` en cualquier capa. 
    // Si Layer es Layer::Zero, busca en la capa cero.
    fn search_single_layer(
        &self,
        q: &T,
        searcher: &mut Searcher,
        layer: Layer<&[Node<M>]>,
        cap: usize,
    ) {

        // Vamos sacando vecinos de la cola de candidatos.
        while let Some(Neighbor { vid, index, .. }) = searcher.candidates.pop() {
            for neighbor in match layer {
                Layer::NonZero(layer) => layer[index as usize].get_neighbors(), // Si es no-cero buscamos en capa no-cero
                Layer::Zero => self.zero[index as usize].get_neighbors(), // Si es cero buscamos en capa cero.
            } {
                // Para cada vecino:
                let node_to_visit = match layer {
                    // Esta parte es para evitar re-visitar nodos, usamos la capa cero,
                    //  ya que sus nodos son consistentes para todas las capas.
                    // Si es capa no cero se obtiene el nodo de la capa zero al que apunta
                    Layer::NonZero(layer) => layer[neighbor as usize].zero_node,
                    // Si es capa cero se obtiene un vecino
                    Layer::Zero => neighbor,
                };

            
                // TODO:¿Bloom filter?
                // Hemos visto el nodo?
                if searcher.seen.insert(node_to_visit) {
                    // Si no lo hemos visto, calcular la distancia del nodo a q
                    let distance = (self.distance_fn)(q, &self.features[node_to_visit as usize]);

                    // Intenta insertar en la cola de cercanos.
                    // Buscar la posicion del primer elemento donde la distancia deja de ser menor o igual a la que acabamos de calcular.
                    let pos = searcher.nearest.partition_point(|n| n.distance <= distance);
                    // Usa partitionpoint por eficiencia ya que así evitamos tener que reordenar la lista de cercanos.

                    if pos != cap {
                        // It was successful. Now we need to know if its full.
                        if searcher.nearest.len() == cap {
                            // In this case remove the worst item.
                            searcher.nearest.pop();
                        }
                        // Añadimos el nuevo elemento.
                        let candidate = Neighbor {
                            index: neighbor as usize,
                            distance,
                        };
                        // Lo insertamos en la posición deseada.
                        searcher.nearest.insert(pos, candidate);
                        // Lo añadimos a la lista de candidatos.
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }

    }

     /// Buscar en la capa zero, simplemente ejecutar search_single_layer con los parametros adecuados.
     fn search_zero_layer(&self, q: &T, searcher: &mut Searcher, cap: usize) {
        self.search_single_layer(q, searcher, Layer::Zero, cap);
    }

    /// Inicia la búsqueda para el siguiente nivel.
    ///
    /// `m` es el número máximo de vecinos a considerar durante la búsqueda.
    fn lower_search(&self, layer: &[Node<M>], searcher: &mut Searcher) {
        // Limpiamos la lista de candidatos para actualizarla.
        searcher.candidates.clear();
        // Solo mantenemos  el primer candidato, como indica el paper original.
        // https://arxiv.org/abs/1603.09320
        // (Ver algoritmo 5 linea 5)
        let &Neighbor { index, distance } = searcher.nearest.first().unwrap();
        searcher.nearest.clear();
        // Bajamos al nodo de la siguiente capa.
        let new_index = layer[index].next_node as usize;
        let candidate = Neighbor {
            index: new_index,
            distance,
        };
        // Guardamos el nodo más cercano en la lista de cercanos y de candidatos
        searcher.nearest.push(candidate);
        searcher.candidates.push(candidate);
    }

    /// Inserta una nueva característica en HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher) -> usize {
        // Obtener el nivel de la característica.
        let level = self.random_level();
        // Si el nivel no es el más alto, cap = ef_construction
        let mut cap = if level >= self.layers.len() {
            self.ef_construction
        } else {
            1
        };

        // Si el índice esta vacío lo añadimos manualmente.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(NeighborNodes {
                neighbors: [!0; M0],
                // M0 es el número de nodos de la capa 0.
            });
            self.features.push(q);

            // Añadir todas las capas donde se encuentra esta feature.
            while self.layers.len() < level {
                
                let node = Node {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: NeighborNodes { neighbors: [!0; M] }, // M es el número de nodos de las capas no cero
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        // Recargar searcher
        self.initialize_searcher(&q, searcher);

        // Realizamos una búsqueda aproximada hasta alzanzar el nivel deseado.
        for ix in (level..self.layers.len()).rev() {
            // Realizar búsqueda aproximada
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Bajamos la búsqueda.
            self.lower_search(&self.layers[ix], searcher);
            cap = if ix == level {
                // Cuando alcanzamos el nivel deseado, actualizar cap para matchear ef_construction.
                self.ef_construction
            } else {
                1
            };
        }

        // Nivel alcanzado, conectamos el nodo a sus vecinos, empleanso
        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            // Buscamos los vecinos de esta capa
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Usar los resultados de la búsqueda para crear el nodo.
            self.create_node(&q, &searcher.nearest, ix + 1);
            // Bajar una capa.
            self.lower_search(&self.layers[ix], searcher);
            cap = self.ef_construction;
        }

        // Conectar el nodo en la capa cero
        self.search_zero_layer(&q, searcher, cap);
        self.create_node(&q, &searcher.nearest, 0);
        // Añadir la feature a la capa cero.
        self.features.push(q);

        // Añadir todos los vectores necesarios para crear el nivel.
        let zero_node = self.zero.len() - 1;
        while self.layers.len() < level {
            let node = Node {
                zero_node,
                next_node: self.layers.last().map(|l| l.len() - 1).unwrap_or(zero_node),
                neighbors: NeighborNodes { neighbors: [!0; M] },
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

     /// Genera un nivel aleatorio
     fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.next_u64() as f64 / core::u64::MAX as f64;
        (-libm::log(uniform) * libm::log(M as f64).recip()) as usize
    }


    // Devuelve el elemento item de  la lista de features.
    pub fn feature(&self, item: usize) -> &T {
        &self.features[item as usize]
    }

    // Obtiene features por capa.
    pub fn layer_feature(&self, level: usize, item: usize) -> &T {
        &self.features[self.layer_item_id(level, item) as usize]
    }

    // Mapea los ids de los elementos de las capas más altas a la capa cero.
    pub fn layer_item_id(&self, level: usize, item: usize) -> usize {
        if level == 0 {
            item
        } else {
            self.layers[level][item as usize].zero_node
        }
    }

    // Número de capas.
    pub fn layers(&self) -> usize {
        self.layers.len() + 1
    }

    // Longitud de la capa 0
    pub fn len(&self) -> usize {
        self.zero.len()
    }
    // Longitud de cualquier capa.
    pub fn layer_len(&self, level: usize) -> usize {
        if level == 0 {
            self.features.len()
        } else if level < self.layers() {
            self.layers[level - 1].len()
        } else {
            0
        }
    }

    // Si la capa cero está vacía, consideramos que el índice está vacío
    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }


    pub fn layer_is_empty(&self, level: usize) -> bool {
        self.layer_len(level) == 0
    }

  

    /// Obtiene la feature de entrada.
    fn entry_feature(&self) -> &T {
        if let Some(last_layer) = self.layers.last() {
            &self.features[last_layer[0].zero_node as usize]
        } else {
            &self.features[0]
        }
    }



    /// Crea un nuevo nodo en un capa dada a partir de sus vecinos.
    fn create_node(&mut self, q: &T, nearest: &[Neighbor], layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            // Obtener el siguiente índice de la capa cero.
            let mut neighbors: [usize; M0] = [!0; M0];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index as usize; // Copiar nearest en neighbors
            }
            // Creamos el struct vecinos
            let node = NeighborNodes { neighbors };
            for neighbor in node.get_neighbors() {

                self.add_neighbor(q, new_index as usize, neighbor, layer);
            }// Insertamos sus vecinos con la función `add_neighbor`.
            // Esta función sirve también para la capa 0.

            // Añadimos el nodo a la capa zero.
            self.zero.push(node);
        } else {

            // Equivalente en el resto de capas.
            let new_index = self.layers[layer - 1].len();
            let mut neighbors: [usize; M] = [!0; M];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let node = Node {
                zero_node: self.zero.len(),
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                },
                neighbors: NeighborNodes { neighbors },
            };
            for neighbor in node.get_neighbors() {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// 
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target_ix: usize, layer: usize) {
        // Obtenemos la feature y los vecinos del target a partir de su índice.
        let (target_feature, target_neighbors) = if layer == 0 {
            (
                &self.features[target_ix],
                &self.zero[target_ix].neighbors[..],
            )
        } else {
            // Si no estamos en capa cero hacemos el mapeo a esta, como de costumbre.
            let target = &self.layers[layer - 1][target_ix];
            (
                &self.features[target.zero_node],
                &target.neighbors.neighbors[..],
            )
        };

        // Obtengo el primer índice de la lista de vecinos donde hay un hueco.
        let empty_point = target_neighbors.partition_point(|&n| n != !0);

        // Si el índice no es la última posición de la lista
        if empty_point != target_neighbors.len() {
            // Añadir el vecino en el punto encontrado.
            if layer == 0 {
                self.zero[target_ix as usize].neighbors[empty_point] = node_ix;
            } else {
                self.layers[layer - 1][target_ix as usize]
                    .neighbors
                    .neighbors[empty_point] = node_ix;
            }
        } else {
            // En cualquier otro caso, hay que encontrar el peor vecino
            let (worst_ix, worst_distance) = target_neighbors
                .iter()
                .enumerate()
                .filter_map(|(ix, &n)| {
                    // Computar la distancia de manera que sea lo mas alta posible si el vecino no se ha rellenado aun
                    if n == !0 {
                        None // usize::MAX  
                    } else {
                        // Calcular la peor distancia
                        let distance = (self.distance_fn)(
                            target_feature,
                            &self.features[if layer == 0 {
                                n
                            } else {
                                self.layers[layer - 1][n].zero_node
                            }],
                        );
                        Some((ix, distance))
                    }
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater))
                .unwrap();

            // Si la distancia mejora la peor distancia, añadimos el nodo en el último lugar
            if (self.distance_fn)(q, target_feature) < worst_distance {
                if layer == 0 {
                    self.zero[target_ix as usize].neighbors[worst_ix] = node_ix;
                } else {
                    self.layers[layer - 1][target_ix as usize]
                        .neighbors
                        .neighbors[worst_ix] = node_ix;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------
// Wrapper para poder integrarlo en el proyecto.


pub type DefaultHNSW<F,R> = Hnsw<F, Vec<f32>, R, 16, 40>;

// IMPLEMENTACIÓN DE VFSANN//
pub struct VFSANNIndex<F, R> 
where
    F: Fn(&Vec<f32>, &Vec<f32>) -> f32,
{
    /// Función de distancia entre dos VFSVector:
    hnsw: DefaultHNSW<F, R>,
    searcher: Searcher, 


}


impl<F,R> VFSANNIndex<F, R> 
where
    F: Fn(&Vec<f32>, &Vec<f32>) -> f32,
    R: RngCore + SeedableRng,
{

    pub fn new(distance_fn: F, ef_construction: Option<usize>) -> Self {

        let candidates = Vec::new();
        let searcher = Searcher::new(candidates);
        let hnsw = DefaultHNSW::new(distance_fn, ef_construction);

        Self {
            hnsw,
            searcher

        }
    }

    pub fn get_searcher(&self) -> Searcher{
        self.searcher.clone()
    }

    pub fn insert_one(&mut self, vfs_vector: VFSVector) -> usize {
        let vf32 = vfs_vector.as_f32_vec();
        let mut searcher = self.get_searcher();
        self.hnsw.insert(vf32, &mut searcher)

    }

    pub fn insert_many(&mut self, vec_list: Vec<VFSVector>){
        for v in vec_list{
            self.insert_one(v);
        }

    }

    pub fn query(&mut self, vfs_vector: &VFSVector) -> Vec<Neighbor> {
        // Inicializar output como un vector mutable
        let mut output = vec![
            Neighbor {
                index: !0,
                distance: 0.0,
            };
            1
        ];
    
        // Convertir el vector a un slice de f32
        let vf32 = vfs_vector.as_f32_vec();
    
        // Inicializar el searcher
        let mut searcher = self.get_searcher();
    
        // Realizar la búsqueda
        let mut result = self.hnsw.nearest(&vf32, 24, &mut searcher, &mut output);

        output
    
       
    }

}


use std::fs::{File, OpenOptions};
use std::io::{self, BufReader,  Read, Write};
use bincode;
use super::vector::{VFSVector}; // Asegúrate de importar correctamente
use super::serializer::{save_vector, load_vectors}; // Funciones de acceso a disco
use super::err::VFSError;
use std::simd::{SupportedLaneCount, LaneCount};
use core::simd::Simd;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

const FLUSH_THRESHOLD: usize = 10; // Número de vectores que se pueden almacenar en memoria antes de flushear la memtable.
const STORAGE_PATH: &str = "data/vectors.dat";
const VFS_STATE_PATH: &str =  "state/vfs_state.bin";

use indexmap::IndexMap; 

#[derive(Debug, Serialize, Deserialize)]
struct VFSState{
    name: String,
    next_id: u64,
    current_offset: usize,
    index_map: BTreeMap<u64, usize>,

}


// Estructura para opciones de reseteo
pub struct ResetOptions {
    pub truncate_data_file: bool,
    pub storage_path: Option<&'static str>,
    pub reset_offset: bool,
    pub new_offset: Option<usize>,
    pub clear_memtable: bool,
    pub clear_indexmap: bool,
    pub reset_id_counter: bool,
    pub new_id_start: Option<u64>,
}

// Implementación con valores predeterminados
impl Default for ResetOptions {
    fn default() -> Self {
        Self {
            truncate_data_file: false,
            storage_path: None,
            reset_offset: true,
            new_offset: Some(0),
            clear_memtable: false,
            clear_indexmap: false,
            reset_id_counter: false,
            new_id_start: None
        }
    }
}

pub struct VFSManager {
    pub name: String,
    index_map: BTreeMap<u64, usize>, // Usamos BTREEMap para asemejar la estructura btree típica de las bases de datos relacionales.
    // Representa un índice id -> offset y perimite realizar búsquedas rápidas por id.
    next_id: u64,
    memtable: IndexMap<u64, VFSVector>, /// Uso indexmap en vez de hashmap por que respeta el orden y unicidad. Funciona mejor
    current_offset: usize,
}

impl VFSManager {
    pub fn new(name: &str) -> Self {

        
        VFSManager {
            name: name.to_string(),
            index_map: BTreeMap::new(),
            next_id: 1,
            memtable: IndexMap::new(),
            current_offset: 0,
        }
    }

    pub fn reset_state(&mut self, options: ResetOptions) -> io::Result<()> {  // Añadido el tipo de retorno
       // Manejar el archivo de datos según las opciones
        if options.truncate_data_file {
            let file_path = options.storage_path.unwrap_or(STORAGE_PATH);
        
            {
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(file_path)?;
             // El archivo se cierra automáticamente al final de este bloque
            }
        
             println!("Archivo de datos truncado en: {}", file_path);
        }
    
        // Resetear el estado interno según las opciones
        if options.reset_offset {
            self.current_offset = options.new_offset.unwrap_or(0);
        }
    
        if options.clear_memtable {
            self.memtable = IndexMap::new();
        }

        if options.clear_indexmap {
            self.index_map = BTreeMap::new();
        }
    
        if options.reset_id_counter {
            self.next_id = options.new_id_start.unwrap_or(1);
        }
    
        
        
        Ok(())  
    }

    fn next_id(&mut self) -> Result<u64, VFSError> {
        let aux = self.next_id;
        self.next_id += 1;
        Ok(aux)
    }


    pub fn get_current_offset(&self) -> usize{
        println!("Current offset is {}", self.current_offset);
        self.current_offset
    }
  

    fn flush_memtable_to_disk(&mut self) -> std::io::Result<()> {
        for (id, vector) in self.memtable.drain(..) {
            let offset = save_vector(&vector, STORAGE_PATH)?;
            self.index_map.insert(id, offset); // Indexar los vectores 
        }
        Ok(())
    }

    pub fn flush_manual(&mut self) -> Result<(), VFSError> {
        self.flush_memtable_to_disk().map_err(|e| VFSError::MemtableError(e.to_string()))?;
        Ok(())
    }

    

    pub fn load_batch(&mut self, count: usize) -> std::io::Result<Vec<VFSVector>> {
        let mut batch: Vec<VFSVector> = Vec::new();
        // P1: Extraer de la memtable solo la cantidad de vectores requerida si es posible
        if !self.memtable.is_empty() {
            // Calcular cuántos elementos vamos a extraer: el mínimo entre el count solicitado y la cantidad en memtable.
            let to_extract = count.min(self.memtable.len());
        
            // Extraer esos elementos sin vaciar toda la memtable.
            // Esto depende del tipo de colección que uses. Si es un Vec, podrías hacer:
            let mem_vectors: Vec<VFSVector> = self.memtable.drain(..to_extract).map(|(_, v)| v).collect();

            // Hacer flush de estos vectores a disco. Se deben guardar a disco ya que si estaban en la memtable significa que no han sido flusheados
            for vector in &mem_vectors {
                let offset = save_vector(&vector, STORAGE_PATH)?;
                self.index_map.insert(vector.id(), offset);
            }

            batch.extend(mem_vectors);
        }

        // P2: Si aun no se alcanzó la cantidad requerida, cargar desde disco.
        if batch.len() < count {
            let needed = count - batch.len();
            let (mut entries, new_offset) = load_vectors(STORAGE_PATH, self.current_offset, needed, None)?;
            self.current_offset = new_offset;
            batch.append(&mut entries);
        }

        Ok(batch)
    }

    fn load_vector_at_offset(&self, offset: usize) -> Result<(VFSVector), VFSError> {
        // Carga un único vector en el offset especificado.
        let (vec, _) = load_vectors(STORAGE_PATH, offset, 1, None)?;
        if vec.len() == 0 {
            println!("No había vectores en ese offset");
            return Err(VFSError::InvalidVector("No vectors at the specified offset".to_string()))
        }
        return Ok(vec[0].clone())
    }

    pub fn get_max_id(&self) -> u64 {
        self.next_id - 1
    }


    // Esto es muy ineficiente, mejorarlo para no hacer la búsqueda lineal.
    pub fn get_vector_by_id(&mut self, id: u64) -> Option<VFSVector> {
        // Paso 1. Mirar si el vector está en la memtable.
        if let Some(vector) = self.memtable.get(&id) {
            println!("Vector encontrado en memtable con ID: {}", id);
            return Some(vector.clone());
        }

        // Paso 2. Comprobamos el índice para no tener que hacer una lectura secuencial.
        if let Some(offset) = self.index_map.get(&id) {
            println!("Vector encontrado en el índice!");
            let vector = self.load_vector_at_offset(*offset).ok()?;
            return Some(vector);
        }
        
        // Paso 3: Oh no, el vector no estaba en la memtable ni en el índice.
        let options = ResetOptions::default(); // resetea el offset poniendolo a 0.
        
        if let Err(e) = self.reset_state(options) {
            println!("Error al resetear el estado: {}", e);
            return None;
        }
        
        // Variable para almacenar el resultado encontrado
        let mut result = None;
         // Búsqueda lineal vector por vector
        loop {
            let vectores = match self.load_batch(1) {
                Ok(v) => v,
                Err(e) => {
                println!("Error al cargar batch: {}", e);
                break;
                }
            };
        
            if vectores.is_empty() {
                println!("Archivo vacío o fin de archivo alcanzado");
                break; // Salir del loop si llegamos al final sin encontrar el ID
            }
        
            // Como cargamos de a uno, solo verificamos el único vector del batch
            let vector = &vectores[0];
        
            if vector.id() == id {
                println!("Vector encontrado con ID: {}", id);
                self.vector_to_memtable(vector.clone());
                result = Some(vector.clone());
                break;
            }
        
            // Si no es el ID buscado, continuamos con el siguiente vector
        }
    

        let restore_options = ResetOptions::default(); 
        if let Err(e) = self.reset_state(restore_options) {
            println!("Advertencia: Error al restaurar el offset original: {}", e);
        }
        
        result

    }

    pub fn get_memtable_size(&self) -> usize {
        self.memtable.len()
    }

    pub fn get_total_vectors_estimate(&self) -> usize {
        self.next_id as usize - 1
    }

    fn vector_to_memtable(&mut self, vector: VFSVector) -> Result<(), VFSError>{
        let id = vector.id();
        self.memtable.insert(id, vector);

        if self.memtable.len() >= FLUSH_THRESHOLD {
            self.flush_memtable_to_disk().map_err(|e| VFSError::MemtableError(e.to_string()))?;
        }
        Ok(())
    }

    // Registra un vector desde Vec
    pub fn register_vector_from_vec(&mut self, data: Vec<f32>, name: &str, tags: Vec<String>) -> Result<u64, VFSError> {
        // Validar datos de entrada
        if data.is_empty() {
            return Err(VFSError::InvalidVector("Vector data cannot be empty".to_string()));
        }

        // Generar un nuevo ID
        let id = self.next_id().map_err(|e| VFSError::IdGenerationError(e.to_string()))?;

        // Crear el vector VFS
        let vfs = VFSVector::from_vec(data, id, name, tags);
        
        // Guardarlo en la memtable
        self.vector_to_memtable(vfs).map_err(|e| VFSError::MemtableError(format!("Error saving vector to memtable: {}", e)))?;
            println!("VFSVector registrado correctamente !!!");
        
        
        Ok(id)
    }

    // Registra un vector desde Simd
    pub fn register_vector_from_simd<const N: usize>(
        &mut self,
        data: Simd<f32, N>,
        name: &str, 
        tags: Vec<String>,
        quantize: bool,
        scale_factor: Option<f32>
    ) -> Result<u64, VFSError>
    where
        LaneCount<N>: SupportedLaneCount,
        {
            
        
        // Generar un nuevo ID
        let id = self.next_id().map_err(|e| VFSError::IdGenerationError(e.to_string()))?;

        // Crear el vector VFS
        let vfs = VFSVector::from_simd(data, id, name, tags, quantize, scale_factor);

        // Guardarlo en la memtable
        self.vector_to_memtable(vfs)
        .map_err(|e| VFSError::MemtableError(format!("Error saving vector to memtable: {}", e)))?;
            println!("VFSVector registrado correctamente !!!");
        
       
        Ok(id)
    }
    
    // Realiza un snapshot del estado actual del manager y lo graba en disco.
    pub fn save_state(&mut self, path: Option<&'static str>) -> Result<(), VFSError> {
        let fpath = path.unwrap_or(VFS_STATE_PATH);
        // Crear el directorio si no existe
        if let Some(parent) = std::path::Path::new(fpath).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let state = VFSState {
            next_id: self.next_id,
            name: self.name.clone(),
            current_offset: self.current_offset,
            index_map: self.index_map.clone(),
            
        };

        let encoded: Vec<u8> = bincode::serialize(&state)
        .map_err(|e| VFSError::SerializationError(e.to_string()))?;


        // Abrimos el archivo
        match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(fpath)
        {
            Ok(mut file) => {
                println!("Archivo abierto correctamente: {}", fpath);
                file.write_all(&encoded)?;
            },
            Err(e) => {
                println!("Error al abrir el archivo '{}': {:?}", fpath, e);
                return Err(VFSError::IoError(e));
            }
        }

        

        // Guardar la memtable
        self.flush_memtable_to_disk().map_err(|e| VFSError::MemtableError(e.to_string()))?;

        Ok(())
    }

    pub fn load_state(&mut self, path: Option<&'static str>) -> Result<(), VFSError> {
        let fpath = path.unwrap_or(VFS_STATE_PATH);
        let mut file = File::open(fpath)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let state: VFSState = bincode::deserialize(&buffer)
        .map_err(|e| VFSError::SerializationError(e.to_string()))?;

        self.next_id = state.next_id;
        self.name = state.name;
        self.index_map = state.index_map;
        self.current_offset = state.current_offset;
        Ok(())
    }
}
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader,  Read, Write};
use bincode;
use super::vector::{VFSVector}; // Asegúrate de importar correctamente
use super::serializer::{save_vector, load_vectors}; // Tus funciones personalizadas
use std::simd::{SupportedLaneCount, LaneCount};
use core::simd::Simd;
use serde::{Serialize, Deserialize};

const FLUSH_THRESHOLD: usize = 1000;
const STORAGE_PATH: &str = "data/vectors.dat";

#[derive(Debug, Serialize, Deserialize)]
struct VFSState{
    name: String,
    next_id: u64,
    current_offset: usize

}


// Estructura para opciones de reseteo
pub struct ResetOptions {
    pub truncate_data_file: bool,
    pub storage_path: Option<&'static str>,
    pub reset_offset: bool,
    pub new_offset: Option<usize>,
    pub clear_memtable: bool,
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
            reset_id_counter: false,
            new_id_start: None
        }
    }
}

pub struct VFSManager {
    pub name: String,
    next_id: u64,
    memtable: HashMap<u64, VFSVector>,
    current_offset: usize,
}

impl VFSManager {
    pub fn new(name: &str) -> Self {

        
        VFSManager {
            name: name.to_string(),
            next_id: 1,
            memtable: HashMap::new(),
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
            self.memtable = HashMap::new();
        }
    
        if options.reset_id_counter {
            self.next_id = options.new_id_start.unwrap_or(1);
        }
    
        
        
        Ok(())  
    }

    fn next_id(&mut self) -> u64 {
        let aux = self.next_id;
        self.next_id += 1;
        aux
    }

    pub fn get_current_offset(&self) -> usize{
        println!("Current offset is {}", self.current_offset);
        self.current_offset
    }
  

    pub fn flush_memtable_to_disk(&mut self) -> std::io::Result<()> {
        for (_, vector) in self.memtable.drain() {
            save_vector(&vector, STORAGE_PATH)?;
        }
        Ok(())
    }

    pub fn flush_manual(&mut self) -> std::io::Result<()> {
        self.flush_memtable_to_disk()
    }

    pub fn load_batch(&mut self, count: usize) -> std::io::Result<Vec<VFSVector>> {
        let (entries, new_offset) = load_vectors(STORAGE_PATH, self.current_offset, count, None)?;
  
        self.current_offset = new_offset;
        Ok(entries)
    }


    // Esto es muy ineficiente, mejorarlo para no hacer la búsqueda lineal.
    pub fn get_vector_by_id(&mut self, id: u64) -> Option<VFSVector> {
        // Paso 1. Mirar si el vector está en la memtable.
        if let Some(vector) = self.memtable.get(&id) {
            println!("Vector encontrado en memtable con ID: {}", id);
            return Some(vector.clone());
        }
        
        // Oh no, el vector no estaba en la memtable.
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

    fn vector_to_memtable(&mut self, vector: VFSVector) -> std::io::Result<()>{
        let id = vector.id();
        self.memtable.insert(id, vector);

        if self.memtable.len() >= FLUSH_THRESHOLD {
            self.flush_memtable_to_disk().expect("Fallo al flushear vectores");
        }
        Ok(())
    }

    // Registra un vector desde Vec
    pub fn register_vector_from_vec(&mut self, data: Vec<f32>, name: &str, tags: Vec<String>) -> u64 {
        
            let id = self.next_id();
            let vfs = VFSVector::from_vec(data, id, name, tags);
            self.vector_to_memtable(vfs).expect("Error guardando el vector en la Memtable");
            println!("VFSVector registrado correctamente !!!");
        
        
        id
    }

    // Registra un vector desde Simd
    pub fn register_vector_from_simd<const N: usize>(
        &mut self,
        data: Simd<f32, N>,
        name: &str, 
        tags: Vec<String>,
        quantize: bool,
        scale_factor: Option<f32>
    ) -> u64
    where
        LaneCount<N>: SupportedLaneCount,
        {
            
        
            let id = self.next_id();
            let vfs = VFSVector::from_simd(data, id, name, tags, quantize, scale_factor);
            self.vector_to_memtable(vfs).expect("Error guardando el vector en la Memtable");
            println!("VFSVector registrado correctamente !!!");
        
       
        id
    }

    pub fn save_state(&mut self, path: &str) -> io::Result<()> {

        // Crear el directorio si no existe
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let state = VFSState {
            next_id: self.next_id,
            name: self.name.clone(),
            current_offset: self.current_offset,
        };

        let encoded: Vec<u8> = bincode::serialize(&state)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;


        // Abrimos el archivo
        match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path) 
        {
            Ok(mut file) => {
                println!("Archivo abierto correctamente: {}", path);
                file.write_all(&encoded)?;
            },
            Err(e) => {
                println!("Error al abrir el archivo '{}': {:?}", path, e);
                return Err(e);
            }
        }

        

        // Guardar la memtable
        self.flush_memtable_to_disk().expect("Fallo al flushear vectores");

        Ok(())
    }

    pub fn load_state(&mut self, path: &str) -> io::Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let state: VFSState = bincode::deserialize(&buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        self.next_id = state.next_id;
        self.name = state.name;
        self.current_offset = state.current_offset;
        Ok(())
    }
}
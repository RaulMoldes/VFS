use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter};
use bincode;
use super::vector::{VFSVector}; // Asegúrate de importar correctamente
use super::serializer::{save_vector, load_vectors}; // Tus funciones personalizadas
use std::simd::{SupportedLaneCount, LaneCount};
use core::simd::Simd;

const FLUSH_THRESHOLD: usize = 1000;
const STORAGE_PATH: &str = "data/vectors.dat";

#[derive(Debug)]
struct VFSState{
    name: String,
    next_id: u64,
    current_offset: usize

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

    fn next_id(&mut self) -> u64 {
        let aux = self.next_id;
        self.next_id += 1;
        aux
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

    pub fn get_memtable_size(&self) -> usize {
        self.memtable.len()
    }

    pub fn get_total_vectors_estimate(&self) -> usize {
        self.next_id as usize - 1
    }

    fn vector_to_memtable(&self, vector: VFSVector) -> std::io::Result<()>{
        let id = vector.id();
        self.memtable.insert(id, vfs);

        if self.memtable.len() >= FLUSH_THRESHOLD {
            self.flush_memtable_to_disk().expect("Fallo al flushear vectores");
        }

    }

    // Registra un vector desde Vec
    pub fn register_vector_from_vec(&mut self, data: Vec<f32>, name: &str, tags: Vec<String>) -> u64 {
        
            let id = self.next_id();
            let vfs = VFSVector::from_vec(data, id, name, tags);
            self.vector_to_memtable(vfs)?;
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
    ) -> Self
    where
        LaneCount<N>: SupportedLaneCount,
        {
            
        
            let id = self.next_id();
            let vfs = VFSVector::from_simd(data, id, name, tags, quantize, scale_factor);
            self.vector_to_memtable(vfs)?;
            println!("VFSVector registrado correctamente !!!");
        
       
        id
    }

    fn save_state(&self, path: &str) -> io::Result<()> {
        let state = VFSState {
            next_id: self.next_id,
            name: self.name.clone(),
            current_offset: self.current_offset,
        };

        let encoded: Vec<u8> = bincode::serialize(&state)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.write_all(&encoded)?;

        // Guardar la memtable
        self.flush_memtable_to_disk().expect("Fallo al flushear vectores");

        Ok(())
    }

    fn load_state(&mut self, path: &str) -> io::Result<()> {
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
#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use core::simd::Simd;
use std::io;

use vfs::vector::VFSVector;
use vfs::storage_manager::VFSManager;

const VFS_STATE_PATH: &str =  "vfs_state.bin";

fn main() -> io::Result<()> {
    // Crear vectores SIMD
    let simd_vector1 = Simd::from_array([1.0, 2.0, 3.0, 4.0]);
    let simd_vector2 = Simd::from_array([1.0, 2.0, 4.0, 4.0]);
    let simd_vector3 = Simd::from_array([1.0, 2.0, 2.0, 4.0]);


    // Inicializar el VFSManager
    let mut manager = VFSManager::new("my_manager");

    // Crear instancias de VFSVector a partir de los vectores SIMD
    manager.register_vector_from_simd(simd_vector1, "SIMD Example 1", vec!["tag1".into()], false, None);
    manager.register_vector_from_simd(simd_vector2, "SIMD Example 2", vec!["tag2".into()], false, None);
    manager.register_vector_from_simd(simd_vector3, "SIMD Example 3", vec!["tag3".into()], false, None);
    manager.register_vector_from_simd(simd_vector3, "SIMD Example 34", vec!["tag3".into()], false, None);
    // Guardar el estado del VFSManager
    manager.save_state(VFS_STATE_PATH)?;

    // Inicializar el nuevo VFSManager
    let mut restored_manager = VFSManager::new("my_other_manager");

    // Cargar el estado del VFSManager desde el archivo
    restored_manager.load_state(VFS_STATE_PATH)?;

    // Cargar los vectores desde el archivo en disco
    
    let vectors = restored_manager.load_batch(2).expect("Error al cargar los vectores");

    // Mostrar información de los vectores cargados
    for vector in vectors.iter() {
        println!("Vector ID: {:?}", vector.id());
        println!("Nombre: {}", vector.metadata().name);
        println!("Tags: {:?}", vector.metadata().tags);
        println!("Valores: {:?}", vector.as_f32_vec());
        println!("---");
    }

    Ok(())
}
#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use core::simd::Simd;
use std::io;

use vfs::vector::VFSVector;
use vfs::storage_manager::VFSManager;
use vfs::rank::{Ranker, SearchType,DistanceMethod};

const VFS_STATE_PATH: &str =  "vfs_state.bin";

fn main() -> io::Result<()> {
    // Crear vectores SIMD
    let simd_vector1 = Simd::from_array([3.0, 7.0, 6.0, 9.0]);
    let simd_vector2 = Simd::from_array([1.0, 2.0, 4.0, 4.9]);
    let simd_vector4 = Simd::from_array([1.1, 2.1, 4.1, 5.0]);
    let simd_vector3 = Simd::from_array([5.0, 7.0, 2.5, 4.7]);
    let simd_vector5 = Simd::from_array([3.0, 7.0, 6.0, 9.0]);
    let simd_vector6 = Simd::from_array([1.0, 2.0, 4.0, 4.9]);
    let simd_vector6 = Simd::from_array([5.0, 7.0, 2.5, 4.7]);

    // Query. El más similar es el simd3.
    let query = VFSVector::from_simd(simd_vector4, 27, "SIMD Example query", vec!["tag1".into()], false, None);

    // Inicializar el VFSManager
    let mut manager = VFSManager::new("my_manager");

    manager.reset_state();
    // Crear instancias de VFSVector a partir de los vectores SIMD
    manager.register_vector_from_simd(simd_vector1, "SIMD Example 1", vec!["tag1".into()], false, None);
    manager.register_vector_from_simd(simd_vector2, "SIMD Example 2", vec!["tag2".into()], false, None);
    manager.register_vector_from_simd(simd_vector3, "SIMD Example 3", vec!["tag3".into()], false, None);
    //manager.register_vector_from_simd(simd_vector4, "SIMD Example 4", vec!["tag4".into()], false, None);
    manager.register_vector_from_simd(simd_vector5, "SIMD Example 5", vec!["tag5".into()], false, None);
    manager.register_vector_from_simd(simd_vector6, "SIMD Example 6", vec!["tag6".into()], false, None);

    // Guardar la memtable en disco.
    manager.flush_manual();

    let mut ranker = Ranker::new(SearchType::Approximate, DistanceMethod::Euclidean, manager);
    let result = match ranker.search(&query, 3, Some(4)) {
        Ok(result_data) => result_data,
        Err(e) => {
            eprintln!("Error en la búsqueda: {}", e);
            return Err(e); // O maneja el error como prefieras
        }
    };

    println!("=== Resultados de la búsqueda ===");
    if result.is_empty() {
    println!("No se encontraron resultados.");
    } else {
    println!("{:<10} {:<15}", "ID", "DISTANCIA");
    println!("{:-<26}", "");  // Línea separadora
    
    for (i, (id, distancia)) in result.iter().enumerate() {
        println!("{:<10} {:<15.6}", id, distancia);
    }
    
    println!("\nTotal de resultados: {}", result.len());
}



    Ok(())
}
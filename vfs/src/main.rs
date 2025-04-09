#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use core::simd::Simd;
use std::io;

use vfs::vector::VFSVector;
use vfs::serializer::{save_vector, load_vectors};
use vfs::rank::{Ranker, SearchType, DistanceMethod};

fn main() -> io::Result<()> {
    // Crear un Simd con un tamaño específico (4 en este caso)
    let simd_vector = Simd::from_array([1.0, 2.0, 3.0, 4.0]);
    let simd_vector2 = Simd::from_array([1.0, 2.0, 4.0, 4.0]);
    let simd_vector3 = Simd::from_array([1.0, 2.0, 2.0, 4.0]);
    // Convertirlo a una instancia de SimdVector::Vec4

    // Crear el VectorEntry a partir del SimdVector
    let entry1 = VFSVector::from_simd(simd_vector, "SIMD Example", vec!["tag1".into()], false, None);

    // Crear el VectorEntry a partir del SimdVector
    let entry2 = VFSVector::from_simd(simd_vector2, "SIMD Example 2", vec!["tag2".into()], false, None);

    // Crear el VectorEntry a partir del SimdVector
    let entry3 = VFSVector::from_simd(simd_vector3, "SIMD 3 Example", vec!["tag1".into()], false, None);



    // Guardar el vector en un archivo
    save_vector(&entry1, "vectors1.bin")?;
    save_vector(&entry2, "vectors1.bin")?;
    save_vector(&entry3, "vectors1.bin")?;

    // Ruta del archivo
    let file_path = "vectors1.bin";
    if let Ok((vectors, offset)) = load_vectors(file_path, 0, 4, Some(50 as usize)) {

        println!("Offset: {}", offset);
        for vector in vectors {
        println!("Vector: {:?}",vector);
        println!("\n\n")
        }
    }


    
    println!("Probando búsqueda exacta.....");
    let simd_vector4 = Simd::from_array([1.1, 2.1, 2.1, 4.1]);
    let query = VFSVector::from_simd(simd_vector4, "SIMD 4 Example", vec!["tag1".into()], false, None);

    let ranker = Ranker::new(SearchType::Approximate, DistanceMethod::Cosine);

    let output = ranker.search(&query, file_path, 2, Some(1024), Some(1));

    //println!("{:?}", output);

    for (vector_box, distance) in output.unwrap().iter() {
        let vector = vector_box.as_ref(); // Obtiene una referencia al Vector dentro del Box
        println!("Vector: {:?}, Distancia: {}", vector, distance);
        println!("\n")
    }

    
    Ok(())
}

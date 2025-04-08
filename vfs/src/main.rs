#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use core::simd::Simd;
use std::io;

use vfs::vector::{Vector, SimdVector};
use vfs::serializer::{save_vector, load_vectors};

fn main() -> io::Result<()> {
    // Crear un Simd con un tamaño específico (4 en este caso)
    let simd = Simd::from_array([1.0, 2.0, 3.0, 4.0]);
    let simd2 = Simd::from_array([1.0, 2.0, 3.0, 4.0]);
    let simd3 = Simd::from_array([1.0, 2.0, 3.0, 4.0]);
    // Convertirlo a una instancia de SimdVector::Vec4
    let simd_vector = SimdVector::Vec4(simd);
    let simd_vector2 = SimdVector::Vec4(simd2);
    let simd_vector3 = SimdVector::Vec4(simd3);
    // Crear el VectorEntry a partir del SimdVector
    let entry1 = Vector::from_simd(simd_vector, "SIMD Example", vec!["tag1".into()]);

    // Crear el VectorEntry a partir del SimdVector
    let entry2 = Vector::from_simd(simd_vector2, "SIMD Example 2", vec!["tag2".into()]);

    // Crear el VectorEntry a partir del SimdVector
    let entry3 = Vector::from_simd(simd_vector3, "SIMD 3 Example", vec!["tag1".into()]);



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

    Ok(())
}

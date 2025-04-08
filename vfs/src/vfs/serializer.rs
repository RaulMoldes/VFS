use std::fs::{OpenOptions, File};
use std::io::{self,Write, Read, Seek, SeekFrom};
use std::option::Option;
use bincode;
use super::vector::Vector;


const fn usize_size() -> usize {
    std::mem::size_of::<usize>()
}

const INT_SIZE: usize = usize_size();
const START_MARKER: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];



// Función para serializar un vector y guardarlo en el archivo que viene dado por path.
// El vector se guarda con un marcador de inicio `START_MARKER`, lo que permite identificar si vamos a leer un vector o no.
// Se guarda también el tamaño del vector para poder avanzar el offset al deserializar.
pub fn save_vector(entry: &Vector, path: &str) -> std::io::Result<()> {
    let bytes = match bincode::serialize(entry){
        Ok(b) => b,
        Err(e) => {
        eprintln!("Error serializando la entrada: {}", e);
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Error serializando la entrada"));
        }
    };


    // Tamaño del vector (en bytes)
    let size = bytes.len() as usize;


    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    

    // Escribir la marca de inicio
    if let Err(e) = file.write_all(&START_MARKER) {
        eprintln!("Error escribiendo la marca de inicio en el archivo: {}", e);
        return Err(e);
    }

    // Escribir el tamaño del vector
    if let Err(e) = file.write_all(&size.to_le_bytes()) {
        eprintln!("Error escribiendo el tamaño del vector en el archivo: {}", e);
        return Err(e);
    }

    // Escribir los datos del vector
    if let Err(e) = file.write_all(&bytes) {
        eprintln!("Error escribiendo los datos del vector en el archivo: {}", e);
        return Err(e);
    }

    Ok(())
}

// Función para cargar un número determinado de vectores en memoria.
// Sirve para cargar el buffer con los vectores que nos interesan.
// count es el número de vectores a cargar.
// offset es la posición en el archivo donde empezamos a leer.
pub fn load_vectors(path: &str, offset: usize, count: usize, buffer_size: Option<usize>) -> io::Result<(Vec<Vector>, usize)> {
    // Abrir el archivo en modo lectura
    let mut file = File::open(path)?;
    let mut entries = Vec::with_capacity(count);
    let mut current_offset = offset;
    let buffer_size = buffer_size.unwrap_or(1024); // Por defecto cargamos 1KB en memoria.

    // Mover el cursor del archivo al offset especificado
    file.seek(SeekFrom::Start(current_offset as u64))?;

   
    let marker_len = START_MARKER.len();
    let mut buffer = vec![0; buffer_size]; // Tamaño del buffer (ajustable según sea necesario)


    // Leer un bloque del archivo
    let bytes_read = file.read(&mut buffer)?;
    println!("Bytes a leer: {}", &bytes_read);

    if bytes_read == 0 {
        return Ok((Vec::new(), current_offset));// Fin del archivo
    }

    let mut cursor = 0; // Cursor para avanzar en el archivo.
    while cursor + marker_len <= bytes_read &&  entries.len() <= count { 
        // Para de leer cuando el buffer esta lleno
        // Buscar la marca de inicio en el buffer
        if &buffer[cursor..cursor + marker_len] == START_MARKER { 
            // Si el primer elemento coincide con la marca de inicio del vector, leer.
            // Saltar la marca de inicio
            println!("Marca de inicio de vector encontrada!");
            cursor += marker_len;

            // Leer el tamaño del vector
            if cursor + INT_SIZE <= bytes_read {
                    
                let size_slice = &buffer[cursor..cursor + INT_SIZE];
                let vector_size = usize::from_le_bytes(size_slice.try_into().unwrap());
                cursor += INT_SIZE;
                println!("Tamaño del vector: {}", vector_size);

                // Verificar que hay suficientes bytes para leer el vector completo
                if cursor + vector_size <= bytes_read {
                    println!("Hay suficientes bytes para leer el vector en el buffer");
                    let vector_slice = &buffer[cursor..cursor + vector_size];
                    match bincode::deserialize::<Vector>(vector_slice) {
                        Ok(entry) => {
                            entries.push(entry);
                           
                        },
                        Err(e) => {
                            eprintln!("Error de deserialización: {:?}", e);
                            cursor = 0;
                            break;
                        },
                    }
                    cursor += vector_size;
                } else {
                    eprintln!("No hay suficientes bytes para leer el vector");
                    cursor = 0;
                    break;
                }
            } else {
                eprintln!("No hay suficientes bytes para leer el tamaño del vector");
                cursor = 0;
                break;
            }
        } else {
                cursor += 1; // Avanzar al siguiente byte
        }
    }
    current_offset += cursor;
    Ok((entries, current_offset))
}
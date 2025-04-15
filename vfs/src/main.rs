#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use std::env;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::io;

use vfs::tcp::{handle_request, ServerState};
use vfs::storage_manager::{VFSManager, ResetOptions};

const DEFAULT_PORT: &str="9001";

fn main() -> io::Result<()> {

    

    let args: Vec<String> = env::args().collect();
    let port = args.get(2).cloned().unwrap_or_else(|| DEFAULT_PORT.to_string());
    let address = format!("127.0.0.1:{}", port);
   // println!("Servidor escuchando en http://127.0.0.1:{}", port);
    let state = Arc::new(Mutex::new(None::<ServerState>));
     // Escuchar en el puerto 7878
     let listener = TcpListener::bind(&address).expect("No se pudo abrir el puerto");

     println!("Servidor escuchando en http://127.0.0.1:{}", port);
 
     // Aceptar conexiones
     for stream in listener.incoming() {
         match stream {
             Ok(stream) => {
                 let state = Arc::clone(&state);
                 thread::spawn(move || {
                     handle_request(stream, state);
                 });
             }
             Err(e) => {
                 eprintln!("Error al aceptar conexión: {}", e);
             }
         }
     }

    Ok(())
}
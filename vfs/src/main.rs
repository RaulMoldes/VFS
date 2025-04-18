#![feature(portable_simd)]
#![allow(dead_code)]
#![allow(unused)]

mod vfs;

use std::env;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::io;
use colored::*;

use vfs::tcp::{handle_request, ServerState};
use vfs::storage_manager::{VFSManager, ResetOptions};

const DEFAULT_PORT: &str="9001";

use colored::*;

pub fn print_welcome_message() {
    println!("{}", "╔══════════════════════════════════════════════════════╗".bright_blue());
    println!("{}", "║                                                      ║".bright_blue());
    println!("{}", "║                   VFS MANAGER API                    ║".bright_yellow().bold());
    println!("{}", "║                                                      ║".bright_blue());
    println!("{}", "║  Endpoints disponibles:                              ║".bright_cyan());
    println!("{}", "║   ➤ POST   /init                                     ║".white());
    println!("{}", "║   ➤ POST   /vectors                                  ║".white());
    println!("{}", "║   ➤ GET    /vectors/<id>                             ║".white());
    println!("{}", "║   ➤ POST   /search                                   ║".white());
    println!("{}", "║   ➤ POST   /flush                                    ║".white());
    println!("{}", "║   ➤ POST   /snapshot                                 ║".white());
    println!("{}", "║   ➤ POST   /restore                                  ║".white());
    println!("{}", "║                                                      ║".bright_blue());
    println!("{}", "║  Contacto: raul.moldes.work@gmail.com                ║".bright_green());
    println!("{}", "║                                                      ║".bright_blue());
    println!("{}", "╚══════════════════════════════════════════════════════╝".bright_blue());
}


fn main() -> io::Result<()> {

    
    print_welcome_message();
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
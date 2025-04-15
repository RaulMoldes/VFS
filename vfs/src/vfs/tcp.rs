// Código para la api TCP que permite interactuar con VFS Manager.

use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::{Value, json};
use core::simd::Simd;

// Importaciones de tus módulos VFS
use super::vector::VFSVector;
use super::err::VFSError;
use super::storage_manager::{VFSManager, ResetOptions};
use super::rank::{Ranker, SearchType, DistanceMethod};

// Estructuras para las peticiones y respuestas
#[derive(Deserialize)]
struct VectorRegisterRequest {
    values: Vec<f32>,
    name: String,
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct InitRequest {
    vector_dimension: usize,
    storage_name: Option<String>, // opcional
    truncate_data: bool,
}

#[derive(Deserialize)]
struct SearchRequest {
    values: Vec<f32>,
    top_k: usize,
    ef_search: Option<usize>,
    search_type: Option<String>,
    distance_method: Option<String>,
}

#[derive(Serialize)]
struct VectorResponse {
    id: u64,
    values: Vec<f32>,
    name: String,
    tags: Vec<String>,
   
}

#[derive(Serialize)]
struct SearchResult {
    id: u64,
    distance: f32,
    vector: Option<VectorResponse>, // Opcional para ahorrar ancho de banda
}

// Estado del servidor
pub struct ServerState {
    manager: VFSManager,
    vector_dimension: usize,
}

// Función para procesar la solicitud HTTP
pub fn handle_request(mut stream: TcpStream, state: Arc<Mutex<Option<ServerState>>>) {
    // Buffer para leer la solicitud
    let mut buffer = [0; 4096];
    
    // Leer la solicitud
    let bytes_read = stream.read(&mut buffer).unwrap_or(0);
    if bytes_read == 0 {
        return;
    }
    
    // Convertir los bytes a string
    let request_str = String::from_utf8_lossy(&buffer[..bytes_read]);
    
    // Parsear la solicitud HTTP
    let request_lines: Vec<&str> = request_str.lines().collect();
    if request_lines.is_empty() {
        return;
    }
    
    // Obtener método y ruta
    let request_parts: Vec<&str> = request_lines[0].split_whitespace().collect();
    if request_parts.len() < 2 {
        return;
    }
    
    let method = request_parts[0];
    let path = request_parts[1];
    
    // Encontrar el cuerpo JSON si existe
    let mut body = String::new();
    let mut found_empty_line = false;
    
    for line in request_lines.iter() {
        if found_empty_line {
            body.push_str(line);
        } else if line.is_empty() {
            found_empty_line = true;
        }
    }
    
    // Procesar la solicitud
    let (status, response_body) = match (method, path) {
        ("GET", p) if p.starts_with("/vectors/") => {
            let id_str = p.trim_start_matches("/vectors/");
            match id_str.parse::<u64>() {
                Ok(id) => get_vector(id, &state),
                Err(_) => (400, json!({"error": "Invalid vector ID"}).to_string()),
            }
        },
        ("POST", "/init") => {
          
            if let Ok(init_request) = serde_json::from_str::<InitRequest>(&body) {

                    println!("Solictud correcta");
                    init_manager(init_request, &state)
                } else {
                    (400, json!({"error": "Invalid JSON for init"}).to_string())
                }
            
        }
        ("POST", "/vectors") => {
            if let Ok(request) = serde_json::from_str::<VectorRegisterRequest>(&body) {
                register_vector(request, &state)
            } else {
                (400, json!({"error": "Invalid JSON request"}).to_string())
            }
        },
        ("POST", "/search") => {
            if let Ok(request) = serde_json::from_str::<SearchRequest>(&body) {
                search(request, &state)
            } else {
                (400, json!({"error": "Invalid JSON request"}).to_string())
            }
        },
        ("POST", "/flush") => {
            flush_memtable(&state)
        },
        ("POST", "/snapshot") => {
            save_state(&state)
        }
        ("POST", "/restore") => {
            load_state(&state)
        }
        _ => (404, json!({"error": "Not found"}).to_string()),
    };
    
    // Construir la respuesta HTTP
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        status_code_to_text(status),
        response_body.len(),
        response_body
    );
    
    // Enviar la respuesta
    let _ = stream.write(response.as_bytes());
    let _ = stream.flush();
}




// Funciones de utilidad
fn status_code_to_text(status: u16) -> &'static str {
    match status {
        200 => "200 OK",
        201 => "201 Created",
        204 => "204 No Content",
        400 => "400 Bad Request",
        404 => "404 Not Found",
        500 => "500 Internal Server Error",
        _ => "200 OK",
    }
}


fn save_state(state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut guard = state.lock().unwrap();
    if let Some(inner_state) = guard.as_mut() {
        match inner_state.manager.save_state(None) {
            Ok(_) => (200, json!({"status": "State saved successfully"}).to_string()),
            Err(e) => (
                500,
                json!({"error": format!("Failed to save state: {}", e)}).to_string(),
            ),
        }
    } else {
        (400, json!({"error": "VFS Manager not initialized"}).to_string())
    }
}


fn load_state(state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut guard = state.lock().unwrap();
    if let Some(inner_state) = guard.as_mut() {
        match inner_state.manager.load_state(None) {
            Ok(_) => (200, json!({"status": "State loaded successfully"}).to_string()),
            Err(e) => (
                500,
                json!({"error": format!("Failed to load state: {}", e)}).to_string(),
            ),
        }
    } else {
        (400, json!({"error": "VFS Manager not initialized"}).to_string())
    }
}


fn flush_memtable(state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut guard = state.lock().unwrap();

    if let Some(inner_state) =  guard.as_mut() {

    match inner_state.manager.flush_manual() {
        Ok(_) => (200, json!({"status": "success", "message": "Memtable flushed to disk"}).to_string()),
        Err(e) => (500, json!({
            "status": "error",
            "error": e.to_string(),
            "error_type": match e {
                VFSError::MemtableError(_) => "memtable_error",
                _ => "internal_error"
            }
        }).to_string())
    }
} else {
    (
        400,
        json!({
            "error": "VFSManager is not initialized"
        })
        .to_string(),
    )
}
}


fn get_vector(id: u64, state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut state_guard = state.lock().unwrap();

    if let Some(inner_state) = state_guard.as_mut() {
    
    if let Some(vector) = inner_state.manager.get_vector_by_id(id) {
        let values = vector.as_f32_vec();
        let response = VectorResponse {
            id: vector.id(),
            values,
            name: vector.metadata().name.clone(),
            tags: vector.metadata().tags.clone(),
        };
        (200, serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()))
    } else {
        (404, json!({"error": "Vector not found"}).to_string())
    }
    } else {
        (
            400,
            json!({
                "error": "VFSManager is not initialized"
            })
            .to_string(),
        )
    }
}

fn init_manager(req: InitRequest, state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut guard = state.lock().unwrap();
    println!("He obtenido el lock");
    if guard.is_some() {
        return (400, json!({"error": "VFSManager is already initialized"}).to_string());
    }

    let mut manager = VFSManager::new(&req.storage_name.unwrap_or_else(|| "default_vfs".into()));

    let reset_options = ResetOptions {
        truncate_data_file: req.truncate_data,
        storage_path: None,
        reset_offset: true,
        new_offset: Some(0),
        clear_memtable: true,
        clear_indexmap: true,
        reset_id_counter: true,
        new_id_start: Some(1),
    };

    manager.reset_state(reset_options);

    *guard = Some(ServerState {
        manager,
        vector_dimension: req.vector_dimension,
    });

    (200, json!({"status": "initialized"}).to_string())
}

fn register_vector(req: VectorRegisterRequest, state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut state_guard = state.lock().unwrap();

    if let Some(inner_state) = state_guard.as_mut() {

    // Validar dimensión del vector
    if req.values.len() != inner_state.vector_dimension {
        return (400, json!({
            "error": format!("Vector dimension mismatch. Expected {}, got {}", 
                            inner_state.vector_dimension, req.values.len())
        }).to_string());
    }
    
    // Crear un vector F32 a partir de los valores
    // Intentar registrar el vector usando la nueva función que devuelve Result
    match inner_state.manager.register_vector_from_vec(
        req.values, 
        &req.name, 
        req.tags.clone()
    ) {
        Ok(id) => {
            // Registro exitoso
            return (201, json!({
                "id": id,
                "status": "success"
            }).to_string());
        },
        Err(e) => {
            // Determinar el código de estado HTTP basado en el tipo de error
            let status_code = match &e {
                VFSError::InvalidVector(_) => 400, // Bad Request
                VFSError::MemtableError(_) => 500, // Internal Server Error
                VFSError::IdGenerationError(_) => 500, // Internal Server Error
                VFSError::IoError(_) => 500, // Internal Server Error
                VFSError::SerializationError(_) => 422, // Unprocessable Entity
                _ => 500, // Internal Server Error por defecto
            };

            // Devolver el error en formato JSON
            (status_code, json!({
            "error": e.to_string(),
            "error_type": match e {
                VFSError::InvalidVector(_) => "invalid_vector",
                VFSError::MemtableError(_) => "storage_error",
                VFSError::IdGenerationError(_) => "id_generation_error",
                VFSError::IoError(_) => "io_error",
                VFSError::SerializationError(_) => "serialization_error",
                _ => "unknown_error",}}).to_string())

            }
        }
    } else {
        (
            400,
            json!({
                "error": "VFSManager is not initialized"
            })
            .to_string(),
        )
    }
            
    
}
    

fn search(req: SearchRequest, state: &Arc<Mutex<Option<ServerState>>>) -> (u16, String) {
    let mut state_guard = state.lock().unwrap();
    
    if let Some(inner_state) = state_guard.as_mut() {
    // Validar dimensión del vector de consulta
    if req.values.len() != inner_state.vector_dimension {
        return (400, json!({
            "error": format!("Query vector dimension mismatch. Expected {}, got {}", 
                            inner_state.vector_dimension, req.values.len())
        }).to_string());
    }
    let id = !0 as u64;
    // Creamos un VFSVector a partir de la solicitud
    let query_vector = VFSVector::from_vec(req.values, id, "Query", vec!["Query".to_string(), "f32".to_string(), "vec".to_string()]);

    
    // Configurar la búsqueda
    let search_type = match req.search_type.as_deref() {
        Some("exact") => SearchType::Exact,
        _ => SearchType::Approximate,
    };
    
    let distance_method = match req.distance_method.as_deref() {
        Some("cosine") => DistanceMethod::Cosine,
        Some("euclidean") => DistanceMethod::Euclidean,
        _ => DistanceMethod::Euclidean,
    };
    
    // Ejecutar búsqueda
    let ef_search = req.ef_search.unwrap_or(6);
    let mut ranker = Ranker::new(search_type, distance_method, Some(ef_search));
    
    // Medir tiempo de consulta
    let start = std::time::Instant::now();
    
    match ranker.search(&query_vector, req.top_k, Some(ef_search), &mut inner_state.manager) {
        Ok(search_results) => {
            let query_time = start.elapsed();
            
            // Convertir resultados a formato JSON
            let results: Vec<SearchResult> = search_results.iter()
                .map(|(id, distance)| {
                    // Opcionalmente, incluir los vectores completos
                    let vector = inner_state.manager.get_vector_by_id(*id).map(|v| {
                        VectorResponse {
                            id: v.id(),
                            values: v.as_f32_vec(),
                            name: v.metadata().name.clone(),
                            tags: v.metadata().tags.clone(),
                            
                        }
                    });
                    
                    SearchResult {
                        id: *id,
                        distance: *distance,
                        vector,
                    }
                })
                .collect();
            
            let response = json!({
                "results": results,
                "query_time_ms": query_time.as_secs_f32() * 1000.0
            });
            
            (200, response.to_string())
        },
        Err(e) => {
            (500, json!({"error": format!("Search error: {}", e)}).to_string())
        }
        }
    } else {
    (
        400,
        json!({
            "error": "VFSManager is not initialized"
        })
        .to_string(),
    )
    }

}

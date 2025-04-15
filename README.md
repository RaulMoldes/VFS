# VFS

VFS es un motor de búsquedas y almacenamiento de vectores escrito en Rust. Empecé a desarrollarlo inspirado por el curso de Carnegie Mellon University Database Group: `Introduction to Database Systems (2025)`

En este curso, Andy Pavlo explica muy detalladamente como funcionan los detalles intrínsecos de los principales motores de almacenamiento que conocemos en la industria. Aunque no se centra específicamente en bases de datos vectoriales, muchas de las técnicas que él explica las he aplicado para desarrollar el proyecto (cache buffer, carga de datos por lotes, índices).

VFS no es ni pretende ser una base de datos vectorial empleada en entornos productivos en ningún caso. Se trata de un proyecto personal orientado a mejorar mi entiendimiento de los sistemas basados en búsqueda vectorial.

# Arquitectura de VFS
![Arquitectura de VFS](imgs/architecture_vfs.png)

* **VFSManager**: es el principal orquestador del motor de almacenamiento. Mantiene el estado del sistema, el cual se guarda periódicamente en disco para evitar la pérdida de datos. Además, es el somponente encargado de cargar vectores en memoria a través de una caché que se va actualizando de forma dinámica, adaptándose a las peticiones del usuario.

* La struct **VFSVector** es otro componente muy importante. Es la principal abstracción de un vector en VFS. Puede contruirse desde un vector Rust (`Vec`) o un vector SIMD (`Simd`), y tiene dos implementaciones posibles:
    - **QuantizedVector**: vector cuantizado y comprimido, más eficiente en entornos limitados.
    - **Vector**: vector de `float32`.

* El módulo **Ranker** es el principal motor de búsquedas. Soporta búsqueda lineal y aproximada. La segunda usa el algoritmo de Hierarchical Navigable Small Worlds, característico de otras bases de datos vectoriales como `Pinecone` o `Qdrant`. Esto permite realizar búsquedas de similitud con mucha precisión incluso aunque el número de vectores almacenados sea elevado. Las medidas de distancia soportadas son:
    - `Euclidean`: distancia euclídea.
    - `Cosine`: distancia coseno.
    - `SimdEuclidean`: euclídea con operaciones SIMD.
    - `SimdCosine`: coseno con operaciones SIMD.

# Características principales:

* **Búsqueda Exacta y Aproximada**: Soporta tanto búsquedas exactas como aproximadas para adaptarse a diferentes necesidades y balances entre precisión y rendimiento.​ Para la búsqueda exacta se usa un mecanismo de ordenamiento de datos por lotes, mientras que para la búsqueda aproximada, se usa el algoritmo `HNSW: Hierarchical Navigable Small World`.

* **Procesamiento por Lotes**: Incluye un buffer pool en memoria integrado y parametrizable para procesar vectores por lotes, permitiendo realizar búsquedas exactas en datasets gigantescos.


* **Operaciones SIMD (Single Instruction, Multiple Data)**: Permite realizar operaciones vectoriales de una forma optimizada usando una feature experimental de `Rust`; la librería `core::Simd` (disponible en `nightly`).

* **Almacenamiento Flexible**: Permite la gestión de vectores en memoria y en disco, adaptándose a diferentes requisitos de almacenamiento y rendimiento.​ Soporta diferentes niveles de precisión, permitiendo almacenar vectores de `float32` para sistemas que requieren una mayor precisión, o vectores cuantizados de `int8`, para aquellos sistemas más limitados en cuanto a recursos se refiere.

VFS es un proyecto en desarrollo activo, y se agradece cualquier contribución o sugerencia para mejorar sus funcionalidades y rendimiento.

# Instrucciones de uso

## Dependencias

Asegúrate de tener instalado en tu equipo local:

- `Rust`: instala Rust siguiendo las instrucciones oficiales en: https://www.rust-lang.org/tools/install
- `RustUp`: RustUp es la herramienta de gestión de versiones de Rust. Puedes instalarla desde: https://rustup.rs/
- `Cargo`: Cargo se instala automáticamente junto con Rust. Para más información sobre Cargo y su instalación, consulta: https://doc.rust-lang.org/cargo/getting-started/installation.html
 
## Iniciar el sevidor:

El servidor local de VFS puede iniciarse de forma sencilla por línea de comandos. Desde la carpeta del proyecto ejecutar:

```bash
cargo run +nightly <port>
```

Asegurarse de usar un puerto válido entre  0 y 65535

## Documentacion de la APi

VFS proporciona una API que permite interactuar con el VFS Manager a través de conexiones TCP usando mensajes HTTP con cuerpos JSON. Cada solicitud se procesa y se devuelve una respuesta HTTP con código de estado y un cuerpo JSON. A continuación se describen los endpoints disponibles:

### Endpoints

1. **POST /init**

* **Descripción**:

Inicializa el estado global del servidor. Este endpoint configura el VFSManager según la dimensión de vector que se desee y otros parámetros de inicialización.

* **Request (JSON):**

```json

{
  "vector_dimension": 4,
  "storage_name": "my_vfs",       // Opcional: nombre del almacenamiento. Si no se proporciona, se usa "default_vfs".
  "truncate_data": true           // Booleano que indica si se debe truncar el archivo de datos.
}
```

* **Respuesta:**

  - 200 OK ```json {  "status": "initialized"}```
  - 400 Bad request si ya esta inicializado: ```json {"error": "VFSManager is already initialized" }``` o si el json enviado es inválido: ```json { "error": "Invalid JSON for init"}```


2. **POST /vectors**

* **Descripción:**

Registra un nuevo vector en el VFS Manager.

* **Request (JSON):**

```json
{
  "values": [1.0, 2.0, 3.0, 4.0],   // Los valores deben coincidir en número con "vector_dimension" establecido previamente.
  "name": "Vector de ejemplo",
  "tags": ["demo", "test"]
}
```

* **Respuesta:**

  - 201 Created: En caso de éxito, se devolverá el ID del vector registrado: ```json {  "id": 1, "status": "success"}```
  - 400 Bad Request: Si la dimensión del vector no coincide: ```json { "error": "Vector dimension mismatch. Expected 4, got N" }```
  - 500 Internal Server Error o 422 Unprocessable Entity: En caso de otros errores, se devolverá un JSON con el error y una clave "error_type" para identificar el tipo específico de error (por ejemplo, "memtable_error", "io_error", etc.).

3. **GET /vectors/<id>**

* **Descripción:**

Obtiene un vector almacenado dado su identificador único.

* **Parámetros en la ruta:**

  - <id>: Un entero que representa el identificador del vector.

* **Respuesta:**

  - 200 OK: Si se encuentra el vector, se devuelve en formato JSON: ```json {  "id": 1, "values": [1.0, 2.0, 3.0, 4.0], "name": "Vector de ejemplo",  "tags": ["demo", "test"]}```
  - 404 Not Found: Si no se encuentra el vector: ```{ "error": "Vector not found" }```
  - 400 Bad Request: Si el ID proporcionado es inválido:: ```{"error": "Invalid vector ID"}```

4. **POST /search**

* **Descripción:**

Realiza una búsqueda de vectores similares en el VFS Manager utilizando un vector de consulta.

* **Request (JSON):**

```json
{
  "values": [1.0, 2.0, 3.0, 4.0],    // El vector de consulta, con la misma dimensión que la inicializada.
  "top_k": 3,                         // Número máximo de resultados a retornar.
  "ef_search": 6,                     // (Opcional) Parámetro para la eficiencia/precisión de la búsqueda.
  "search_type": "approximate",       // (Opcional) "exact" o "approximate" (por defecto se usa approximate).
  "distance_method": "euclidean"      // (Opcional) Puede ser "euclidean" o "cosine". Por defecto se usa euclidean.
}
```

* **Respuesta:**

  - 200 OK: Devuelve un JSON con un arreglo de resultados y el tiempo de consulta: ```{ "results": [ { "id": 1, "distance": 0.123456, "vector": {  "id": 1,"values": [1.0, 2.0, 3.0, 4.0],"name":"Vector de ejemplo", "tags": ["demo", "test"]} }, { ... }  ],"query_time_ms": 12.34} ```
  - 400 Bad Request: Si la dimensión del vector de consulta no coincide: ````{"error": "Query vector dimension mismatch. Expected 4, got N"}```
  - 500 Internal Server Error: Si ocurre algún error durante la búsqueda: ```{"error": "Search error: <detalle del error>"}```

5. **POST /flush**

* **Descripción:**

Fuerza el "flush" manual de la memtable del VFSManager, escribiendo los datos en disco.

* **Request:**

No requiere cuerpo JSON.

**Respuesta:**
  - 200 OK: En caso de éxito: ```{"status": "success", "message": "Memtable flushed to disk"}```
  - 500 Internal Server Error: Si ocurre un error al realizar el flush: ```{"status": "error","error": "<mensaje de error>","error_type": "memtable_error" }```
  - 400 Bad Request: Si el VFSManager no ha sido inicializado: ```{"error": "VFSManager is not initialized"}```

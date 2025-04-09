# VFS
Motor de búsqueda y almacenamiento de vectores escrito en Rust.


Características principales:

* **Búsqueda Exacta y Aproximada**: Soporta tanto búsquedas exactas como aproximadas para adaptarse a diferentes necesidades y balances entre precisión y rendimiento.​ Para la búsqueda exacta se usa un mecanismo de ordenamiento de datos por lotes, mientras que para la búsqueda aproximada, se usa el algoritmo `HNSW: Hierarchical Navigable Small World`.

* **Procesamiento por Lotes**: Incluye un buffer pool en memoria integrado y parametrizable para procesar vectores por lotes, permitiendo realizar búsquedas exactas en datasets gigantescos.

* **Interfaz Intuitiva**: Proporciona una API sencilla y clara, facilitando su integración en diversos proyectos y aplicaciones.​

* **Operaciones SIMD (Single Instruction, Multiple Data)**: Permite realizar operaciones vectoriales de una forma optimizada usando una feature experimental de `Rust`; la librería `core::Simd` (disponible en `nightly`).

* **Almacenamiento Flexible**: Permite la gestión de vectores en memoria y en disco, adaptándose a diferentes requisitos de almacenamiento y rendimiento.​ Soporta diferentes niveles de precisión, permitiendo almacenar vectores de `float32` para sistemas que requieren una mayor precisión, o vectores cuantizados de `int8`, para aquellos sistemas más limitados en cuanto a recursos se refiere.

Este proyecto está en desarrollo activo, y se agradece cualquier contribución o sugerencia para mejorar sus funcionalidades y rendimiento.

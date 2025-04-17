## Inicializar el manager:

```bash
curl -X POST http://127.0.0.1:9001/init      -H "Content-Type: application/json"      -d '{
           "vector_dimension": 4,
           "storage_name": "my_vfs",
           "truncate_data": true,
           "quantize":treu,
         }'
```

##  Generar datos dummy

```bash
#!/bin/bash
{
URL="http://127.0.0.1:9001/vectors"
DIM=4

for i in {1..100}; do
  # Generar 4 valores float aleatorios entre 0 y 1
  VALUES=$(for j in $(seq 1 $DIM); do awk -v seed=$RANDOM 'BEGIN { srand(seed); printf("%.4f", rand()); }'; echo -n ", "; done | sed 's/, $//')

  # Construir el JSON
  JSON=$(cat <<EOF
{
  "values": [ $VALUES ],
  "name": "dummy_vector_$i",
  "tags": ["dummy", "generated"]
}
EOF
)

  # Enviar la solicitud
  curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$JSON" > /dev/null
  echo "Vector $i enviado."
done

echo "Todos los vectores fueron enviados."
}

```

## Flushear memtable a disco

```bash
curl -X POST http://127.0.0.1:9001/flush
```

## Búsqueda aproximada:

```bash
curl -X POST http://127.0.0.1:9001/search      -H "Content-Type: application/json"      -d '{
           "values": [1.0, 2.0, 3.0, 4.0],
           "top_k": 3,
           "ef_search": 6,
           "search_type": "approximate",
           "distance_method": "euclidean"
         }'
```


## Búsqueda exacta:

```bash
curl -X POST http://127.0.0.1:9001/search      -H "Content-Type: application/json"      -d '{
           "values": [1.0, 2.0, 3.0, 4.0],
           "top_k": 3,
           "ef_search": 6,
           "search_type": "exact",
           "distance_method": "euclidean" 
         }'
```

## Buscar por id

```bash
curl -X GET http://127.0.0.1:9001/vectors/1

```
## Guardar y recargar el estado
```bash
curl -X POST http://127.0.0.1:9001/state/save
curl -X POST http://127.0.0.1:9001/state/load
```


# bin/bash

#!/bin/bash

set -euo pipefail  # Falla si hay errores, variables no definidas o errores en pipes

# Ir a la raíz del proyecto (donde está este script)
cd "$(dirname "$0")"

# Verificar que existe la carpeta vfs
if [ ! -d "vfs" ]; then
  echo "❌ Error: No se encontró la carpeta 'vfs'. Asegúrate de estar en la raíz del proyecto."
  exit 1
fi

# Iniciar servidor desde carpeta vfs
echo "🚀 Iniciando servidor VFS en segundo plano desde ./vfs..."
cd vfs
cargo run +nightly 9001 > ../server.log 2>&1 &
SERVER_PID=$!
cd ..

echo "🔧 PID del servidor: $SERVER_PID"

# Esperar a que el servidor esté disponible
echo "⏳ Esperando a que el servidor esté listo en http://localhost:9001..."
for i in {1..15}; do
  if curl -s http://localhost:9001 >/dev/null; then
    echo "✅ Servidor disponible. Ejecutando tests..."
    break
  fi
  sleep 3
  if [ "$i" -eq 15 ]; then
    echo "❌ El servidor no respondió a tiempo. Abortando."
    kill $SERVER_PID || true
    exit 1
  fi
done

API_URL="http://localhost:9001"  # Cambia si tu API corre en otro host/puerto

# Función para ejecutar una solicitud y validar código de estado esperado
function test_endpoint() {
  local method=$1
  local endpoint=$2
  local data=$3
  local expected_status=$4
  local test_name=$5
  local server_pid=$6

  echo -e "\n===> $test_name"

  if [ "$method" = "GET" ]; then
    response=$(curl -s -w "%{http_code}" -o tmp_response.json -X GET "$API_URL$endpoint")
  else
    response=$(curl -s -w "%{http_code}" -o tmp_response.json -X "$method" "$API_URL$endpoint" -H "Content-Type: application/json" -d "$data")
  fi

  cat tmp_response.json | jq .

  if [ "$response" -ne "$expected_status" ]; then
    echo "❌ Error: Se esperaba código HTTP $expected_status pero se recibió $response"
    sleep 4
    echo "Matando servidor..."
    kill -9 $server_pid
    exit 1
  else
    echo "✅ Éxito: Código HTTP $response como se esperaba"
  fi
}

# 1. Inicializar
test_endpoint POST "/init" '{
  "vector_dimension": 4,
  "storage_name": "my_vfs",
  "truncate_data": true
}' 200 "1. Inicializando VFSManager" $SERVER_PID

# 2. Insertar vector válido
test_endpoint POST "/vectors" '{
  "values": [1.0, 2.0, 3.0, 4.0],
  "name": "Vector de ejemplo",
  "tags": ["demo", "test"]
}' 201 "2. Insertando vector válido" $SERVER_PID

# 3. Insertar vector con dimensión inválida
test_endpoint POST "/vectors" '{
  "values": [1.0, 2.0],
  "name": "Vector inválido",
  "tags": ["error"]
}' 400 "3. Insertar vector con dimensión incorrecta" $SERVER_PID

# 4. Obtener vector por ID
test_endpoint GET "/vectors/1" "" 200 "4. Obtener vector con ID 1" $SERVER_PID

# 5. Buscar vector similar
test_endpoint POST "/search" '{
  "values": [1.0, 2.0, 3.0, 4.0],
  "top_k": 3,
  "ef_search": 6,
  "search_type": "approximate",
  "distance_method": "euclidean"
}' 200 "5. Buscar vectores similares" $SERVER_PID

# 6. Flush manual
test_endpoint POST "/flush" '' 200 "6. Ejecutar flush manual" $SERVER_PID

# 7. Guardar snapshot
test_endpoint POST "/snapshot" '' 200 "7. Guardar estado (snapshot)" $SERVER_PID

# 8. Restaurar snapshot
test_endpoint POST "/restore" '' 200 "8. Restaurar estado (snapshot)" $SERVER_PID

# Limpieza
rm -f tmp_response.json

echo -e "\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE"

echo "Matando el servidor..." 

kill $SERVER_PID


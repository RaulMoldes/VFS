import requests
import random
import time
import matplotlib.pyplot as plt
import itertools
import argparse
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:9001"
VECTOR_DIM = 16
POST_VECTOR_COUNT = 100
NUM_QUERIES_GET = 10
TOP_K = 25
RANGES_GET_SEARCH = [10, 20, 30, 35, 45, 50, 75, 90, 200,350, 500,750, 1000]
QUANTIZED = True
USE_SIMD = True
FIGS_FOLDER = '../imgs/quantized/' if QUANTIZED else '../imgs/dense/'
DISTANCE_FN = "cosine"
LOG_FILE = f"log/vf_benchmarks{datetime.now().isoformat(timespec='seconds')}.json"


def init_vfs(quantized = QUANTIZED):
    payload = {
        "vector_dimension": VECTOR_DIM,
        "storage_name": "vfs_benchmark",
        "truncate_data": True,
        "quantize": quantized
    }
    r = requests.post(f"{BASE_URL}/init", json=payload)
    print("Init:", r.status_code, r.json())

def generate_random_vector():
    return [round(random.uniform(-10, 10), 4) for _ in range(VECTOR_DIM)]


def get_vector_by_id(vector_id):
    start = time.time()
    r = requests.get(f"{BASE_URL}/vectors/{vector_id}")
    end = time.time()
    duration = (end - start) * 1000  # ms
    if r.status_code != 200:
        print(f"❌ Error al hacer GET /vectors/{vector_id}: {r.text}")
    return duration

def insert_vectors(n):
    inserted_ids = []
    for i in range(n):
        payload = {
            "values": generate_random_vector(),
            "name": f"Vec_{i}",
            "tags": ["benchmark"]
        }
        r = requests.post(f"{BASE_URL}/vectors", json=payload)
        if r.status_code != 201:
            print(f"❌ Error insertando vector {i}: {r.text}")
        else:
            inserted_ids.append(r.json()["id"])

    return inserted_ids

def search(query_vector, search_type, distance_method = DISTANCE_FN):
    payload = {
        "values": query_vector,
        "top_k": 5,
        "ef_search": 100,
        "search_type": search_type,
        "distance_method": distance_method
    }
    start = time.time()
    r = requests.post(f"{BASE_URL}/search", json=payload)
    end = time.time()
    duration = (end - start) * 1000  # ms
    if r.status_code != 200:
        print(f"❌ Search {search_type} failed:", r.text)
    return duration

######## BENCHMARKS ########

def run_benchmark_post(n = POST_VECTOR_COUNT):
    times = []
    for i in range(n):
        payload = {
            "values": generate_random_vector(),
            "name": f"Vector {i}",
            "tags": ["benchmark"]
        }
        start = time.time()
        r = requests.post(f"{BASE_URL}/vectors", json=payload)
        end = time.time()
        duration = (end - start) * 1000  # ms
        times.append(duration)

        if r.status_code != 201:
            print(f"[ERROR] Vector {i} → {r.status_code}: {r.text}")
        else:
            print(f"[{i+1}/{POST_VECTOR_COUNT}] POST /vectors: {duration:.2f} ms")
    return times

def run_benchmark_search(count, n, use_simd = USE_SIMD, distance_fn = DISTANCE_FN) -> dict:
    print(f"\n🚀 Benchmarking con {n} vectores...")
    
    insert_vectors(count)
    query = generate_random_vector()
    times = {}

    times['Exact (SISD)'] = search(query, "exact", distance_method=distance_fn)
    times['Aprox (SISD)'] = search(query, "approximate", distance_method=distance_fn)

    print(f"  🔎 Tiempo exacto (sisd): {times['Exact (SISD)']:.2f} ms")
    print(f"  🔍 Tiempo aproximado (sisd): {times['Aprox (SISD)']:.2f} ms")

    if use_simd:
        times['Exact (SIMD)'] = search(query, "exact", distance_method=f"simd_{distance_fn}")
        times['Aprox (SIMD)'] = search(query, "approximate", distance_method=f"simd_{distance_fn}")

    print(f"  🔎 Tiempo exacto (simd): {times['Exact (SIMD)']:.2f} ms")
    print(f"  🔍 Tiempo aproximado (simd): {times['Aprox (SIMD)']:.2f} ms")
   

    return times




def run_benchmark_get(n,num_queries=NUM_QUERIES_GET):
    print(f"\n🚀 Benchmarking GET /vectors/<id> con {n} vectores insertados...")
    
    ids = insert_vectors(n)
    #print(ids)
    sampled_ids = random.choices(ids, k=num_queries)
    times = []

    for vid in sampled_ids:
        t = get_vector_by_id(vid)
        times.append(t)

    avg_time = sum(times) / len(times)
    print(f"  📈 Tiempo promedio: {avg_time:.2f} ms para {num_queries} consultas")
    return avg_time




######## PLOTS ########

def plot_post_results(times, fig_location = f"{FIGS_FOLDER}post_benchmark.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(times, marker='o', linestyle='-', color='dodgerblue')
    plt.title("Tiempos de respuesta: POST /vectors")
    plt.xlabel("Número de vector")
    plt.ylabel("Tiempo (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_location)
    plt.show()



def plot_search_results(results, use_simd = USE_SIMD, fig_location = f"{FIGS_FOLDER}search_benchmark.png"):
    labels = [str(n) for n in results.keys()]

    x = list(range(len(labels)))
    width = 0.2  # ancho más angosto para acomodar las 4 barras
    exact_times = [v['Exact (SISD)'] for v in results.values()]
    aprox_times = [v['Aprox (SISD)'] for v in results.values()]
    if use_simd:
        exact_simd_times = [v['Exact (SIMD)'] for v in results.values()]
        aprox_simd_times = [v['Aprox (SIMD)'] for v in results.values()]

    

    plt.figure(figsize=(12, 6))
    # Dibujar barras con los nuevos colores
    plt.bar([i - 3*width/4 for i in x], exact_times, width, label='Exact (SISD)', color='#FF7F0E')
    if use_simd:
        plt.bar([i - width/4 for i in x], exact_simd_times, width, label='Exact (SIMD)', color='#D62728')
    plt.bar([i + width/4 for i in x], aprox_times, width, label='Approximate (SISD)', color='#1F77B4')
    if use_simd:
        plt.bar([i + 3*width/4 for i in x], aprox_simd_times, width, label='Approximate (SIMD)', color='#2CA02C')
    plt.xticks(x, labels)
    plt.ylabel("Tiempo (ms)")
    plt.xlabel("Número de vectores insertados")
    plt.title("Comparación de búsqueda: Exacta vs Aproximada con operaciones simd o normales")
    plt.legend()
    plt.grid(axis="y", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_location)
    plt.show()


def plot_get_results(results, figs_location = f"{FIGS_FOLDER}get_benchmark.png"):
    sizes = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', linestyle='-', color='mediumseagreen')
    plt.xlabel("Número de vectores insertados")
    plt.ylabel("Tiempo promedio de GET /vectors/<id> (ms)")
    plt.title("Benchmark de acceso por ID en VFS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figs_location)
    plt.show()

def log_execution(content, log_file = LOG_FILE):
    with open(log_file, "w") as f:
        json.dump(content, f, indent=4)

def main(benchmark):

    print("\n" + "="*60)
    print("🚀 Iniciando benchmarks con la siguiente configuración:")
    print("="*60)
    print(f"📍 Base URL         : {BASE_URL}")
    print(f"📏 Dimensión Vector : {VECTOR_DIM}")
    print(f"📦 Vectores a insertar: {POST_VECTOR_COUNT}")
    print(f"🔍 Consultas GET     : {NUM_QUERIES_GET}")
    print(f"🎯 Rango de búsqueda : {RANGES_GET_SEARCH}")
    print(f"🧠 Cuantización      : {'Activada ✅' if QUANTIZED else 'Desactivada ❌'}")
    print(f"⚙️ SIMD              : {'Usando SIMD 🧬' if USE_SIMD else 'Sin SIMD 🚫'}")
    print(f"📁 Carpeta de figuras: {FIGS_FOLDER}")
    print(f"📐 Distancia usada   : {DISTANCE_FN}")
    print("="*60 + "\n")
    
    log = {
        "base_url": BASE_URL,
    "vector_dim": VECTOR_DIM,
    "post_vector_count": POST_VECTOR_COUNT,
    "num_queries_get": NUM_QUERIES_GET,
    "ranges_get_search": RANGES_GET_SEARCH,
    "quantized": QUANTIZED,
    "use_simd": USE_SIMD,
    "figs_folder": FIGS_FOLDER,
    "distance_fn": DISTANCE_FN,
    "datetime": datetime.now().isoformat(timespec='seconds'),
    "method": benchmark
    }

    init_vfs(quantized=QUANTIZED)

    insert_sizes = RANGES_GET_SEARCH
    test_sizes =  list(itertools.accumulate(insert_sizes))
    results = {}
    
    if benchmark == 'GET':
        for num_vect, size in zip(insert_sizes, test_sizes):
            
            result = run_benchmark_get(num_vect, NUM_QUERIES_GET)
            results[size] = result

        print("\n📊 Resultados:", results)
        plot_get_results(results)
    
    elif benchmark == 'SEARCH':
        for num_vect, size in zip(insert_sizes, test_sizes):
            result = run_benchmark_search(count = num_vect,n = size , use_simd = USE_SIMD, distance_fn = DISTANCE_FN)
            results[size] = result

        print("\n📊 Resultados:", results)
        plot_search_results(results)
    
    elif benchmark == 'POST':
       times = run_benchmark_post(n = POST_VECTOR_COUNT)
       results['times'] = times
       plot_post_results(times)

    
    
    else:
        raise ValueError("Tipo de benchmark no permitido. Los tipos permitidos son GET, POST o SEARCH")

    ## Guardar la ejecución
    log['results'] = results
    log_execution(content=log, log_file = LOG_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark",help="Tipo de benchmark a ejecutar: GET, POST, SEARCH")

    args= parser.parse_args()
    main(args.benchmark)
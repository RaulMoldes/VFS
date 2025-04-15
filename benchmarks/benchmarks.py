import requests
import random
import time
import matplotlib.pyplot as plt
import itertools
import argparse

BASE_URL = "http://127.0.0.1:9001"
VECTOR_DIM = 4
TEST_VECTOR_COUNT = 500

def init_vfs():
    payload = {
        "vector_dimension": VECTOR_DIM,
        "storage_name": "vfs_benchmark",
        "truncate_data": True
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

def search(query_vector, search_type):
    payload = {
        "values": query_vector,
        "top_k": 3,
        "ef_search": 6,
        "search_type": search_type,
        "distance_method": "euclidean"
    }
    start = time.time()
    r = requests.post(f"{BASE_URL}/search", json=payload)
    end = time.time()
    duration = (end - start) * 1000  # ms
    if r.status_code != 200:
        print(f"❌ Search {search_type} failed:", r.text)
    return duration

######## BENCHMARKS ########

def run_benchmark_post():
    times = []
    for i in range(TEST_VECTOR_COUNT):
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
            print(f"[{i+1}/{TEST_VECTOR_COUNT}] POST /vectors: {duration:.2f} ms")
    return times

def run_benchmark_search(count, n):
    print(f"\n🚀 Benchmarking con {n} vectores...")
    
    insert_vectors(count)
    query = generate_random_vector()

    time_exact = search(query, "exact")
    time_approx = search(query, "approximate")

    print(f"  🔎 Tiempo exacto: {time_exact:.2f} ms")
    print(f"  🔍 Tiempo aproximado: {time_approx:.2f} ms")

    return (time_exact, time_approx)



def run_benchmark_get(n,num_queries=10):
    print(f"\n🚀 Benchmarking GET /vectors/<id> con {n} vectores insertados...")
    
    ids = insert_vectors(n)
    print(ids)
    sampled_ids = random.choices(ids, k=num_queries)
    times = []

    for vid in sampled_ids:
        t = get_vector_by_id(vid)
        times.append(t)

    avg_time = sum(times) / len(times)
    print(f"  📈 Tiempo promedio: {avg_time:.2f} ms para {num_queries} consultas")
    return avg_time


######## PLOTS ########

def plot_post_results(times):
    plt.figure(figsize=(12, 6))
    plt.plot(times, marker='o', linestyle='-', color='dodgerblue')
    plt.title("Tiempos de respuesta: POST /vectors")
    plt.xlabel("Número de vector")
    plt.ylabel("Tiempo (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../imgs/benchmark_post_vectors.png")
    plt.show()

def plot_search_results(results):
    labels = [str(n) for n in results.keys()]
    exact_times = [v[0] for v in results.values()]
    approx_times = [v[1] for v in results.values()]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], exact_times, width, label='Exact', color='orange')
    plt.bar([i + width/2 for i in x], approx_times, width, label='Approximate', color='deepskyblue')
    plt.xticks(x, labels)
    plt.ylabel("Tiempo (ms)")
    plt.xlabel("Número de vectores insertados")
    plt.title("Comparación de búsqueda: Exacta vs Aproximada")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("../imgs/search_benchmark_comparison.png")
    plt.show()


def plot_get_results(results):
    sizes = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', linestyle='-', color='mediumseagreen')
    plt.xlabel("Número de vectores insertados")
    plt.ylabel("Tiempo promedio de GET /vectors/<id> (ms)")
    plt.title("Benchmark de acceso por ID en VFS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../imgs/get_vector_by_id_benchmark.png")
    plt.show()


def main(benchmark):

    init_vfs()

    insert_sizes = [10, 20, 30, 35, 45, 50, 75, 90, 200,350, 500,750, 1000]
    test_sizes =  list(itertools.accumulate(insert_sizes))
    results = {}
    
    if benchmark == 'GET':
        for num_vect, size in zip(insert_sizes, test_sizes):
            print(num_vect)
            result = run_benchmark_get(num_vect)
            results[size] = result

        print("\n📊 Resultados:", results)
        plot_get_results(results)
    
    elif benchmark == 'SEARCH':
        for num_vect, size in zip(insert_sizes, test_sizes):
            result = run_benchmark_search(num_vect, size)
            results[size] = result

        print("\n📊 Resultados:", results)
        plot_search_results(results)
    
    elif benchmark == 'POST':
       times = run_benchmark_post()
       plot_post_results(times)
    
    else:
        raise ValueError("Tipo de benchmark no permitido. Los tipos permitidos son GET, POST o SEARCH")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark",help="Tipo de benchmark a ejecutar: GET, POST o SEARCH")

    args= parser.parse_args()
    main(args.benchmark)
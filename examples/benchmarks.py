import requests
import random
import time
import json

BASE_URL = "http://127.0.0.1:9001"

VECTOR_DIM = 4
TEST_VECTOR_COUNT = 50

def init_vfs():
    payload = {
        "vector_dimension": VECTOR_DIM,
        "storage_name": "benchmark_test",
        "truncate_data": True
    }
    r = requests.post(f"{BASE_URL}/init", json=payload)
    print("Init:", r.status_code, r.json())

def generate_random_vector():
    return [round(random.uniform(-10, 10), 4) for _ in range(VECTOR_DIM)]

def benchmark_post_vectors():
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
            print("Error saving vector:", r.status_code, r.text)
    return times

def benchmark_search():
    payload = {
        "values": generate_random_vector(),
        "top_k": 5
    }
    start = time.time()
    r = requests.post(f"{BASE_URL}/search", json=payload)
    end = time.time()
    duration = (end - start) * 1000
    return duration, r.status_code

def benchmark_get_vector(vector_id):
    start = time.time()
    r = requests.get(f"{BASE_URL}/vectors/{vector_id}")
    end = time.time()
    return (end - start) * 1000, r.status_code

def run_benchmark():
    print("Inicializando VFS...")
    init_vfs()

    print(f"Iniciando benchmark de POST /vectors con {TEST_VECTOR_COUNT} vectores...")
    post_times = benchmark_post_vectors()
    print(f"→ Tiempo promedio de POST /vectors: {sum(post_times)/len(post_times):.2f} ms")

    print("Benchmarking /search...")
    search_time, status = benchmark_search()
    print(f"→ /search respondió en {search_time:.2f} ms con status {status}")

    print("Benchmarking /vectors/<id> para id=1...")
    get_time, status = benchmark_get_vector(1)
    print(f"→ /vectors/1 respondió en {get_time:.2f} ms con status {status}")

if __name__ == "__main__":
    run_benchmark()
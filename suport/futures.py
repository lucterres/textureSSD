import time
from concurrent.futures import ThreadPoolExecutor

def minha_funcao():
    print("Iniciando...")
    time.sleep(2)
    print("Finalizando...")

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(minha_funcao) for _ in range(2)]

    for future in futures:
        future.result()  # Espera cada uma terminar
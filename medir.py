import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import chatbot_logic as c


def medir(nombre, f):
    t = time.perf_counter()
    r = f()
    print(f"{nombre}: {time.perf_counter() - t:.2f} s")
    return r


medir("arranque (cliente + modelo)", c.precalentar_recursos)
print(medir("clasificador", lambda: c.clasificar_con_confianza("Me caí en la cocina y me duele la pierna")))
r = medir("Gemini texto", lambda: c.generar_respuesta_chatbot([{"role": "user", "content": "Hola"}]))
print("Respuesta de Gemini:", r)
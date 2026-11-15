import time
import tomllib
from pathlib import Path

from google import genai
from google.genai import types

raiz = Path(__file__).resolve().parent
llave = tomllib.load(open(raiz / ".streamlit" / "secrets.toml", "rb"))["GEMINI_API_KEY"]
cliente = genai.Client(api_key=llave, http_options=types.HttpOptions(timeout=60000))

print("Modelos 'flash' disponibles para tu llave:")
try:
    for m in cliente.models.list():
        if "flash" in m.name:
            print("  ", m.name)
except Exception as e:
    print("  No se pudo listar:", str(e)[:200])

for modelo in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"):
    t = time.perf_counter()
    try:
        r = cliente.models.generate_content(model=modelo, contents="Responde solo: hola")
        print(f"\n{modelo}: OK en {time.perf_counter() - t:.1f} s -> {r.text.strip()[:60]}")
    except Exception as e:
        print(f"\n{modelo}: ERROR tras {time.perf_counter() - t:.1f} s -> {str(e)[:200]}")
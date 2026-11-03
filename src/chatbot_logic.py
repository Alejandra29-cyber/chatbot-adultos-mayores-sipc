import re
import os
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai
import streamlit as st

def conectar_gemini():
    try:
        llave_secreta = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=llave_secreta)
    except Exception as e:
        return None

def limpiar_texto_natural(texto):
    texto_limpio = re.sub(r'[\*\#_`]', '', texto)
    return texto_limpio.strip()

def generar_respuesta_chatbot(mensajes, audio_bytes=None):
    cliente = conectar_gemini()
    
    sistema_reglas = (
        "Eres un asistente virtual amigable, conversacional y muy servicial para adultos mayores. "
        "Responde de forma natural, cálida y directa a lo que el usuario te diga. "
        "REGLA DE ORO: No des advertencias médicas ni menciones a doctores a menos que el usuario esté describiendo un dolor físico real. "
        "REGLA ESTRICTA DE FORMATO: Nunca utilices asteriscos, negritas, cursivas ni símbolos markdown. Solo texto plano."
    )
    
    if cliente is not None:
        intentos = 3
        for intento in range(intentos):
            try:
                if audio_bytes is not None:
                    if hasattr(audio_bytes, "read"):
                        bytes_data = audio_bytes.read()
                    else:
                        bytes_data = audio_bytes

                    temp_audio_path = "temp_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(bytes_data)

                    try:
                        audio_file = cliente.files.upload(file=temp_audio_path)
                        
                        prompt_audio = [
                            audio_file,
                            sistema_reglas + "\n\nEscucha este mensaje de voz del usuario, comprende lo que dice y respóndele de forma muy cálida, empática y breve en texto plano:"
                        ]
                        
                        respuesta = cliente.models.generate_content(
                            model='gemini-3.8-flash',
                            contents=prompt_audio,
                        )
                        
                        resultado = limpiar_texto_natural(respuesta.text)
                        
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                            
                        return resultado
                        
                    except Exception as ex_audio:
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                        return f"No se pudo procesar el audio: {ex_audio}"

                
                historial_prompt = sistema_reglas + "\n\n"
                for m in mensajes:
                    if "[Mensaje de voz enviado]" not in m["content"] and m["content"].strip():
                        rol = "Usuario" if m["role"] == "user" else "Asistente"
                        historial_prompt += f"{rol}: {m['content']}\n"
                
                historial_prompt += "Asistente: "

                respuesta = cliente.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=historial_prompt,
                )
                return limpiar_texto_natural(respuesta.text)
                
            except Exception as error:
                if "503" in str(error) and intento < intentos - 1:
                    time.sleep(2)
                    continue
                else:
                    return f"Ocurrió un detalle al conectar con Gemini: {error}"
    else:
        return "Atención: No se encontró configurada la llave de Gemini en el archivo secrets.toml."

def entrenar_baseline():
    ruta = Path(__file__).parent / 'data' / 'mensajes_sinteticos.csv'
    if not ruta.exists():
        ruta = Path('data/mensajes_sinteticos.csv')
        
    df = pd.read_csv(ruta)
    X = df['mensaje'].tolist()
    y = df['categoria'].tolist()
    
    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode')),
        ('regresion_logistica', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    modelo.fit(X, y)
    return modelo

def clasificar_mensaje_usuario(texto_usuario):
    try:
        modelo = entrenar_baseline()
        return modelo.predict([texto_usuario])[0]
    except Exception:
        return "CHARLA_CASUAL"

def guardar_registro_interaccion(tipo_entrada, contenido, clasificacion, nivel_riesgo, alerta):
    ruta_archivo = Path(__file__).parent / 'data' / 'historial_interacciones.json'
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    
    nueva_interaccion = {
        "id_interaccion": f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "tipo_entrada": tipo_entrada,
        "contenido_transcrito": contenido,
        "clasificacion_modelo_baseline": clasificacion,
        "nivel_riesgo": nivel_riesgo,
        "alerta_generada": alerta
    }
    
    registros = []
    if ruta_archivo.exists():
        try:
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                registros = json.load(f)
        except json.JSONDecodeError:
            registros = []
            
    registros.append(nueva_interaccion)
    
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=4)
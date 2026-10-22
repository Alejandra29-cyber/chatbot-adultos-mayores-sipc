import re
import os
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
            return f"Ocurrió un detalle al conectar con Gemini: {error}"
    else:
        return "Atención: No se encontró configurada la llave de Gemini en el archivo secrets.toml."
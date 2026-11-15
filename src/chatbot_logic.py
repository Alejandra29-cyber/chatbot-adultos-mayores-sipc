import re
import os
import sys
import json
import time
import uuid
import shutil
import logging
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google import genai
from google.genai import types
import streamlit as st

logger = logging.getLogger(__name__)


# Constantes del sistema

SISTEMA_REGLAS = (
    "Eres un asistente virtual amigable, conversacional y muy servicial para adultos mayores. "
    "Responde de forma natural, cálida y directa a lo que el usuario te diga. "
    "REGLA DE ORO: No des advertencias médicas ni menciones a doctores a menos que el usuario esté describiendo un dolor físico real. "
    "REGLA ESTRICTA DE FORMATO: Nunca utilices asteriscos, negritas, cursivas ni símbolos markdown. Solo texto plano."
)

MODELO_PRINCIPAL = 'gemini-3.8-flash'
MODELO_RESPALDO = 'gemini-3.5-flash'   # solo si Google rechaza el principal por saturación; poner None para desactivarlo
MIN_TIMEOUT_MS = 10000                 # Google rechaza (400) cualquier límite por llamada menor a 10 s
TIMEOUT_PRINCIPAL_MS = 10000           # límite para gemini-3.8-flash: si se queda esperando, pasa al respaldo
TIMEOUT_RESPALDO_MS = 15000            # el respaldo (o el principal, si no hay respaldo) tiene más tiempo
PAUSA_MODELO_PRINCIPAL_SEG = 60        # tras un rechazo, durante este tiempo se usa directo el respaldo (sin esperar el fallo)

VENTANA_HISTORIAL = 12          # últimos mensajes enviados a Gemini (menos tokens = respuesta más rápida)

# Escala de riesgo (1-5). Umbrales sobre la probabilidad de URGENCIA_CRISIS que entrega la Regresión Logística.
# Calibrados con el dataset sintético actual; conviene recalibrarlos cuando el dataset crezca.
UMBRAL_DUDA = 0.30              # MALESTAR_LEVE con esta probabilidad de crisis sube a nivel 3
UMBRAL_CONFIANZA_CRISIS = 0.45  # URGENCIA_CRISIS con menos confianza que esto queda en nivel 4 (alerta igualmente)
UMBRAL_ALERTA = 4               # nivel >= 4 dispara alerta inmediata a enfermería
PAUSA_GEMINI_SEG = 20           # tras un fallo de Gemini, los mensajes siguientes usan respuesta predefinida durante este tiempo
TIMEOUT_CRISIS_MS = 10000       # en crisis Gemini tiene un solo intento de máximo 10 s; si no responde, se usa la respuesta predefinida
VENTANA_ALERTA_SEG = 600        # no se repite la alerta a enfermería de una misma conversación durante 10 minutos
UMBRAL_SIMILITUD = 0.60         # similitud mínima para reutilizar una respuesta modelo; si no, se usa una respuesta genérica segura

AVISO_ENFERMERIA = "Estoy aquí contigo. Ya avisé de inmediato al equipo de enfermería de la residencia y una enfermera va en camino."
AVISO_ENFERMERIA_PREVIO = "Estoy aquí contigo. El equipo de enfermería ya fue avisado y una enfermera va en camino."
CIERRE_CRISIS = "Quédate donde estás y respira despacio; yo me quedo contigo hasta que lleguen."
TEXTO_ALERTA_FALLIDA = (
    "Estoy aquí contigo. No pude avisar al personal de forma automática, así que por favor pide ayuda "
    "en voz alta o usa el timbre de llamada que tengas cerca. Respira despacio, yo me quedo contigo."
)
TEXTO_AUDIO_NO_ENTENDIDO = (
    "No logré escucharte bien. ¿Podrías repetir tu mensaje o escribirlo aquí abajo? "
    "Si necesitas ayuda urgente, pide apoyo al personal de la residencia."
)
RESPUESTAS_GENERICAS = {
    "CHARLA_CASUAL": "Con mucho gusto te acompaño. Cuéntame, ¿cómo te sientes hoy?",
    "MALESTAR_LEVE": "Gracias por contármelo, estoy aquí contigo. Cuéntame un poco más de cómo te sientes; si algo empeora, avisa al personal de enfermería.",
    "URGENCIA_CRISIS": f"{AVISO_ENFERMERIA} {CIERRE_CRISIS}",
}

_PREFIJOS_ERROR_GEMINI = (
    "Ocurrió un detalle al conectar con Gemini",
    "No se pudo procesar el audio",
    "Atención: No se encontró configurada",
)

_CANDADO_ARCHIVOS = threading.Lock()



# Rutas 
@lru_cache(maxsize=1)
def _dir_data():
    base = Path(__file__).resolve().parent
    for candidato in (base.parent / 'data', base / 'data', Path.cwd() / 'data'):
        if (candidato / 'mensajes_sinteticos.csv').exists():
            return candidato
    return base.parent / 'data'



# Gemini

@st.cache_resource
def conectar_gemini():
    try:
        llave_secreta = st.secrets["GEMINI_API_KEY"]
        return genai.Client(
            api_key=llave_secreta,
            http_options=types.HttpOptions(
                timeout=10000,                                          # si Gemini no responde en 10 s, se usa la respuesta de respaldo
                retry_options=types.HttpRetryOptions(attempts=1),       # sin reintentos internos del SDK (los controlamos nosotros)
            ),
        )
    except Exception as e:
        return None

def limpiar_texto_natural(texto):
    texto_limpio = re.sub(r'[\*\#_`]', '', texto)
    return texto_limpio.strip()

_principal_pausado_hasta = 0.0

def _es_saturacion(error):
    texto = str(error)
    return any(c in texto for c in ("503", "504", "429", "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED"))

def _generar_contenido(cliente, contents, json_salida=False, timeout_ms=None, un_solo_intento=False):
    """Llama a gemini-3.8-flash. Si Google lo rechaza o no responde, usa el modelo de respaldo y lo avisa en la terminal.
    timeout_ms fija el mismo límite para ambos modelos; un_solo_intento evita probar el segundo modelo (casos urgentes)."""
    global _principal_pausado_hasta

    def _config(ms):
        datos = {"http_options": types.HttpOptions(timeout=max(ms, MIN_TIMEOUT_MS))}
        if json_salida:
            datos["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**datos)

    t_principal = timeout_ms or (TIMEOUT_PRINCIPAL_MS if MODELO_RESPALDO else TIMEOUT_RESPALDO_MS)
    t_respaldo = timeout_ms or TIMEOUT_RESPALDO_MS
    if not (MODELO_RESPALDO and time.time() < _principal_pausado_hasta):
        try:
            return cliente.models.generate_content(model='gemini-3.8-flash', contents=contents, config=_config(t_principal))
        except Exception as error:
            if not MODELO_RESPALDO or not _es_saturacion(error) or un_solo_intento:
                if MODELO_RESPALDO and _es_saturacion(error):
                    _principal_pausado_hasta = time.time() + PAUSA_MODELO_PRINCIPAL_SEG
                raise
            _principal_pausado_hasta = time.time() + PAUSA_MODELO_PRINCIPAL_SEG
            print(f"[{datetime.now():%H:%M:%S}] AVISO {MODELO_PRINCIPAL} no disponible ({str(error)[:90]}) "
                  f"-> se usa {MODELO_RESPALDO} por {PAUSA_MODELO_PRINCIPAL_SEG} s", file=sys.stderr, flush=True)
    return cliente.models.generate_content(model=MODELO_RESPALDO, contents=contents, config=_config(t_respaldo))

def generar_respuesta_chatbot(mensajes, audio_bytes=None, timeout_ms=None):
    cliente = conectar_gemini()

    sistema_reglas = SISTEMA_REGLAS

    if cliente is not None:
        # Audio: una sola llamada que transcribe y responde (ver transcribir_y_responder_audio)
        if audio_bytes is not None:
            return transcribir_y_responder_audio(audio_bytes)["respuesta"]

        # timeout_ms (opcional): límite corto por llamada y sin reintentos, para casos urgentes
        intentos = 1 if (timeout_ms or MODELO_RESPALDO) else 3   # con respaldo ya hay un segundo intento
        for intento in range(intentos):
            try:
                historial_prompt = sistema_reglas + "\n\n"
                for m in mensajes[-VENTANA_HISTORIAL:]:
                    if "[Mensaje de voz enviado]" not in m["content"] and m["content"].strip():
                        rol = "Usuario" if m["role"] == "user" else "Asistente"
                        historial_prompt += f"{rol}: {m['content']}\n"

                historial_prompt += "Asistente: "

                respuesta = _generar_contenido(cliente, historial_prompt, timeout_ms=timeout_ms, un_solo_intento=bool(timeout_ms))
                return limpiar_texto_natural(respuesta.text)

            except Exception as error:
                if "503" in str(error) and intento < intentos - 1:
                    time.sleep(0.8 * (intento + 1))
                    continue
                else:
                    return f"Ocurrió un detalle al conectar con Gemini: {error}"
    else:
        return "Atención: No se encontró configurada la llave de Gemini en el archivo secrets.toml."

def _extraer_json_audio(texto):
    """Devuelve (transcripcion, respuesta) a partir de la salida JSON de Gemini, tolerando formatos imperfectos."""
    texto = texto or ""
    datos = {}
    try:
        datos = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        coincidencia = re.search(r'\{.*\}', texto, re.S)
        if coincidencia:
            try:
                datos = json.loads(coincidencia.group(0))
            except json.JSONDecodeError:
                datos = {}
    if not isinstance(datos, dict):
        datos = {}
    transcripcion = str(datos.get("transcripcion", "") or "").strip()
    respuesta = str(datos.get("respuesta", "") or "").strip()
    if not respuesta and not datos and texto.strip() and not texto.lstrip().startswith(("{", "[")):
        respuesta = texto.strip()          # el modelo ignoró el formato y respondió en texto plano
    return transcripcion, limpiar_texto_natural(respuesta)

def transcribir_y_responder_audio(audio_bytes, mime_type="audio/wav"):
    """Una sola llamada a Gemini (audio en línea, sin subir archivos): transcribe lo que dijo el usuario
    y genera la respuesta empática. Devuelve {"transcripcion", "respuesta", "ok"}."""
    cliente = conectar_gemini()
    if cliente is None:
        return {"transcripcion": "", "ok": False,
                "respuesta": "Atención: No se encontró configurada la llave de Gemini en el archivo secrets.toml."}

    if hasattr(audio_bytes, "read"):
        audio_bytes = audio_bytes.read()

    contenido = [
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        SISTEMA_REGLAS
        + "\n\nEscucha este mensaje de voz del usuario. Responde ÚNICAMENTE con un objeto JSON con dos campos: "
        '"transcripcion" (lo que dijo el usuario, palabra por palabra, en español, sin comentarios; cadena vacía si no se entiende) y '
        '"respuesta" (tu respuesta muy cálida, empática y breve, en texto plano sin símbolos markdown).',
    ]

    intentos = 1 if MODELO_RESPALDO else 3
    for intento in range(intentos):
        try:
            resp = _generar_contenido(cliente, contenido, json_salida=True)
            transcripcion, respuesta = _extraer_json_audio(resp.text)
            return {"transcripcion": transcripcion, "respuesta": respuesta, "ok": bool(respuesta)}
        except Exception as error:
            if ("503" in str(error) or "UNAVAILABLE" in str(error)) and intento < intentos - 1:
                time.sleep(0.8 * (intento + 1))
                continue
            return {"transcripcion": "", "ok": False,
                    "respuesta": f"No se pudo procesar el audio: {error}"}



# Clasificador baseline (TF-IDF + Regresión Logística)

@st.cache_resource
def _cargar_dataset():
    df = pd.read_csv(_dir_data() / 'mensajes_sinteticos.csv', encoding='utf-8-sig')
    if 'respuesta' not in df.columns:      
        df['respuesta'] = ""
    df['respuesta'] = df['respuesta'].fillna("")
    return df

@st.cache_resource
def entrenar_baseline():
    df = _cargar_dataset()
    X = df['mensaje'].tolist()      # solo mensaje y categoria entrenan el modelo;
    y = df['categoria'].tolist()    # la columna respuesta es base de conocimiento, no entra al ajuste

    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode')),
        ('regresion_logistica', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    modelo.fit(X, y)
    return modelo

@st.cache_resource
def _indice_respuestas():
    modelo = entrenar_baseline()
    df = _cargar_dataset()
    tfidf = modelo.named_steps['tfidf']
    return tfidf, tfidf.transform(df['mensaje']), df

def clasificar_con_confianza(texto_usuario):
    """Devuelve (categoria, probabilidad_de_URGENCIA_CRISIS) con una sola predicción."""
    if not texto_usuario or not texto_usuario.strip():
        return "SIN_CLASIFICAR", 0.0
    try:
        modelo = entrenar_baseline()
        probas = modelo.predict_proba([texto_usuario])[0]
        clases = list(modelo.classes_)
        categoria = clases[int(np.argmax(probas))]
        p_crisis = float(probas[clases.index("URGENCIA_CRISIS")]) if "URGENCIA_CRISIS" in clases else 0.0
        return categoria, p_crisis
    except Exception:
        logger.exception("Fallo al clasificar el mensaje")
        return "SIN_CLASIFICAR", 0.0

def clasificar_mensaje_usuario(texto_usuario):
    categoria, _ = clasificar_con_confianza(texto_usuario)
    return "CHARLA_CASUAL" if categoria == "SIN_CLASIFICAR" else categoria

def respuesta_modelo(categoria, texto):
    """Respuesta empática del dataset más parecida al mensaje (misma categoría). Se usa si Gemini no responde."""
    try:
        tfidf, matriz, df = _indice_respuestas()
        similitudes = (matriz @ tfidf.transform([texto]).T).toarray().ravel()   # TF-IDF ya normalizado: producto = coseno
        similitudes = np.where((df['categoria'] == categoria).to_numpy(), similitudes, -1.0)
        mejor = int(similitudes.argmax())
        if similitudes[mejor] >= UMBRAL_SIMILITUD and df['respuesta'].iloc[mejor]:
            return df['respuesta'].iloc[mejor]
    except Exception:
        logger.exception("Fallo al buscar respuesta modelo")
    return RESPUESTAS_GENERICAS.get(categoria, RESPUESTAS_GENERICAS["MALESTAR_LEVE"])

def precalentar_recursos():
    """Carga cliente Gemini, dataset y modelo al abrir la app para que el primer mensaje no pague el arranque."""
    conectar_gemini()
    _indice_respuestas()



# DIKW: Evaluación de riesgo -> Decisión -> Acción -> Retroalimentación

def evaluar_riesgo(categoria, p_crisis):
    """Conocimiento -> nivel de riesgo 1 a 5."""
    if categoria == "URGENCIA_CRISIS":
        return 5 if p_crisis >= UMBRAL_CONFIANZA_CRISIS else 4
    if categoria == "MALESTAR_LEVE":
        return 3 if p_crisis >= UMBRAL_DUDA else 2
    if categoria == "CHARLA_CASUAL":
        return 1
    return 3    # SIN_CLASIFICAR: ante la duda, marcar para revisión humana

def decidir_protocolo(nivel_riesgo):
    """Decisión: qué protocolo corresponde al nivel de riesgo."""
    if nivel_riesgo >= UMBRAL_ALERTA:
        return "Protocolo C (riesgo alto o crisis): activar alerta prioritaria inmediata al área de enfermería y contener emocionalmente al usuario"
    if nivel_riesgo == 3:
        return "Protocolo B2 (riesgo moderado): contención emocional y marcar la interacción para revisión de enfermería en la próxima ronda, sin alerta inmediata"
    if nivel_riesgo == 2:
        return "Protocolo B (malestar leve): acompañamiento empático y registro para seguimiento"
    return "Protocolo A (sin riesgo): acompañamiento conversacional normal"

def _nuevo_id():
    return f"int_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

def _ahora_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def _escribir_json_atomico(ruta, datos):
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)
    os.replace(tmp, ruta)

def _notificar_webhook(url, alerta):
    try:
        peticion = urllib.request.Request(
            url, data=json.dumps(alerta, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(peticion, timeout=5)
    except Exception:
        logger.exception("No se pudo notificar al canal externo de enfermería (la alerta sigue en el log clínico)")

def alertar_enfermeria(id_interaccion, nivel_riesgo, categoria, contenido, tipo_entrada):
    """Acción: registra una alerta prioritaria en el log clínico de enfermería (data/alertas_enfermeria.log,
    una línea JSON por alerta) y, si existe ENFERMERIA_WEBHOOK_URL en secrets, la reenvía en segundo plano.
    Devuelve el dict de la alerta, o None si no pudo registrarse."""
    ahora = datetime.now()
    alerta = {
        "id_alerta": f"ALR-{ahora:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4].upper()}",
        "timestamp": _ahora_iso(),
        "area_destino": "Enfermería",
        "prioridad": "CRITICA" if nivel_riesgo >= 5 else "ALTA",
        "estado": "PENDIENTE_DE_ATENCION",
        "id_interaccion": id_interaccion,
        "nivel_riesgo": nivel_riesgo,
        "clasificacion_emocional": categoria,
        "tipo_entrada": tipo_entrada,
        "contenido_transcrito": contenido,
    }
    try:
        ruta = _dir_data() / 'alertas_enfermeria.log'
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with _CANDADO_ARCHIVOS:
            with open(ruta, "a", encoding="utf-8") as f:
                f.write(json.dumps(alerta, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("No se pudo registrar la alerta a enfermería")
        return None

    try:
        url = st.secrets.get("ENFERMERIA_WEBHOOK_URL")
    except Exception:
        url = None
    if url:
        threading.Thread(target=_notificar_webhook, args=(url, alerta), daemon=True).start()
    return alerta



# Persistencia en tiempo real (historial acumulativo)

def _leer_historial(ruta):
    if not ruta.exists() or ruta.stat().st_size == 0:
        return []
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        if isinstance(datos, list):
            return datos
    except (json.JSONDecodeError, OSError):
        pass
    try:    # nunca descartar datos ilegibles: se respaldan antes de empezar de nuevo
        shutil.copy(ruta, ruta.with_name(f"{ruta.stem}.corrupto_{datetime.now():%Y%m%d_%H%M%S}{ruta.suffix}"))
    except OSError:
        pass
    return []

def guardar_registro_interaccion(tipo_entrada, contenido, clasificacion, nivel_riesgo, alerta,
                                 decision="", accion="", respuesta_chatbot="", retroalimentacion="",
                                 id_interaccion=None):
    ruta_archivo = _dir_data() / 'historial_interacciones.json'
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

    nueva_interaccion = {
        "id_interaccion": id_interaccion or _nuevo_id(),
        "timestamp": _ahora_iso(),
        "tipo_entrada": tipo_entrada,
        "contenido_transcrito": contenido,
        "clasificacion_emocional": clasificacion,
        "nivel_riesgo": nivel_riesgo,
        "alerta_generada": alerta,
        "decision": decision,
        "accion": accion,
        "respuesta_chatbot": respuesta_chatbot,
        "retroalimentacion": retroalimentacion,
    }

    with _CANDADO_ARCHIVOS:
        registros = _leer_historial(ruta_archivo)
        registros.append(nueva_interaccion)
        _escribir_json_atomico(ruta_archivo, registros)
    return nueva_interaccion

def registrar_inicio_conversacion(saludo):
    """Registra la apertura de la conversación (turno 0) en el historial."""
    try:
        guardar_registro_interaccion(
            tipo_entrada="texto",
            contenido="[Inicio de conversación]",
            clasificacion="CHARLA_CASUAL",
            nivel_riesgo=1,
            alerta=False,
            decision="Protocolo A (sin riesgo): abrir la conversación con un saludo de bienvenida",
            accion="Saludo inicial del asistente",
            respuesta_chatbot=limpiar_texto_natural(saludo),
            retroalimentacion="Conversación iniciada; en espera del primer mensaje del usuario",
        )
    except Exception:
        logger.exception("No se pudo registrar el inicio de la conversación")



# Pipeline completo de una interacción

def _es_error_gemini(texto):
    return not texto or texto.startswith(_PREFIJOS_ERROR_GEMINI)

_gemini_pausado_hasta = 0.0

def _avisar_terminal(contexto, detalle):
    """El error de Gemini va a la terminal, nunca al chat del usuario."""
    global _gemini_pausado_hasta
    _gemini_pausado_hasta = time.time() + PAUSA_GEMINI_SEG
    print(f"[{datetime.now():%H:%M:%S}] AVISO Gemini no disponible ({contexto}): {str(detalle)[:400]} "
          f"-> se usa respuesta predefinida", file=sys.stderr, flush=True)

def _gemini_texto(mensajes, urgente=False):
    """Respuesta de Gemini para texto, o "" si no está disponible (entonces se usa la predefinida)."""
    if time.time() < _gemini_pausado_hasta:
        return ""    # Gemini falló hace poco: no se hace esperar al usuario
    respuesta = generar_respuesta_chatbot(mensajes, timeout_ms=TIMEOUT_CRISIS_MS if urgente else None)
    if _es_error_gemini(respuesta):
        _avisar_terminal("texto", respuesta)
        return ""
    return respuesta

def _alerta_activa():
    """Alerta reciente de esta misma conversación (evita avisar varias veces a enfermería por la misma emergencia)."""
    try:
        alerta = st.session_state.get("alerta_activa")
    except Exception:
        return None
    if alerta and time.time() - alerta["t"] < VENTANA_ALERTA_SEG:
        return alerta
    return None

def _recordar_alerta(id_alerta):
    try:
        st.session_state["alerta_activa"] = {"id": id_alerta, "t": time.time()}
    except Exception:
        pass

def procesar_interaccion(tipo_entrada, mensajes, texto=None, audio_bytes=None):
    """Dato -> Información -> Conocimiento -> Decisión -> Acción -> Retroalimentación.
    Devuelve la respuesta en texto plano que se muestra al usuario."""
    id_interaccion = _nuevo_id()
    respuesta_llm = ""
    audio_ok = True

    # DATO: lo que entra (texto escrito o audio)
    if tipo_entrada == "audio":
        resultado = transcribir_y_responder_audio(audio_bytes)        # una sola llamada: transcribe y responde
        contenido = resultado["transcripcion"]
        respuesta_llm = resultado["respuesta"] if resultado["ok"] else ""
        audio_ok = resultado["ok"] and bool(contenido)
        if not resultado["ok"]:
            _avisar_terminal("audio", resultado["respuesta"])
    else:
        contenido = (texto or "").strip()

    # CLASIFICACIÓN (Regresión Logística) y nivel de riesgo
    categoria, p_crisis = clasificar_con_confianza(contenido)
    nivel = evaluar_riesgo(categoria, p_crisis)

    # DECISIÓN: protocolo según el nivel
    decision = decidir_protocolo(nivel)

    # ACCIÓN: la alerta sale ANTES de esperar a Gemini en el flujo de texto, para no depender de su latencia
    alerta_info = None
    alerta_previa = None
    if nivel >= UMBRAL_ALERTA:
        alerta_previa = _alerta_activa()
        if alerta_previa is None:
            alerta_info = alertar_enfermeria(id_interaccion, nivel, categoria, contenido, tipo_entrada)
            if alerta_info:
                _recordar_alerta(alerta_info["id_alerta"])

    # Gemini contiene al usuario; en crisis tiene un máximo de 5 s y, si falla, se usa la respuesta predefinida
    if tipo_entrada != "audio":
        respuesta_llm = _gemini_texto(mensajes, urgente=nivel >= UMBRAL_ALERTA)

    # Respuesta final + registro de acción y retroalimentación
    if tipo_entrada == "audio" and not audio_ok:
        contenido = contenido or "[Audio no transcrito]"
        decision = "Sin transcripción confiable: pedir al usuario que repita o escriba y dejar la interacción para revisión humana"
        accion = "Se solicitó repetir el mensaje; interacción marcada para revisión del personal"
        retro = "Pendiente de revisión humana: no se pudo transcribir el audio"
        respuesta_final = TEXTO_AUDIO_NO_ENTENDIDO
    elif nivel >= UMBRAL_ALERTA:
        if alerta_info:
            accion = (f"Alerta prioritaria {alerta_info['id_alerta']} ({alerta_info['prioridad']}) registrada en el log clínico "
                      "de enfermería y contención emocional verbal por el asistente")
            retro = f"Alerta {alerta_info['id_alerta']} en estado PENDIENTE_DE_ATENCION; ciclo abierto hasta que enfermería confirme la atención"
            respuesta_final = (f"{AVISO_ENFERMERIA} {respuesta_llm} {CIERRE_CRISIS}" if respuesta_llm
                               else respuesta_modelo(categoria, contenido))
        elif alerta_previa:
            accion = (f"Alerta previa {alerta_previa['id']} sigue activa (menos de 10 minutos); no se generó otra para evitar duplicados; "
                      "contención emocional verbal por el asistente")
            retro = f"Seguimiento de la alerta {alerta_previa['id']}, pendiente de atención por enfermería; ciclo abierto"
            respuesta_final = (f"{AVISO_ENFERMERIA_PREVIO} {respuesta_llm} {CIERRE_CRISIS}" if respuesta_llm
                               else f"{AVISO_ENFERMERIA_PREVIO} {CIERRE_CRISIS}")
        else:
            accion = "Falló el registro de la alerta; se pidió al usuario solicitar ayuda al personal cercano y se le contuvo verbalmente"
            retro = "Alerta NO registrada: requiere revisión inmediata del personal"
            respuesta_final = TEXTO_ALERTA_FALLIDA
    else:
        respuesta_final = respuesta_llm or respuesta_modelo(categoria, contenido)
        if nivel == 3:
            accion = "Acompañamiento verbal con contención emocional; interacción marcada para revisión en la próxima ronda de enfermería"
            retro = "Pendiente de revisión por enfermería en la próxima ronda; ciclo abierto con seguimiento"
        elif nivel == 2:
            accion = "Acompañamiento verbal empático; registro para seguimiento"
            retro = "Seguimiento rutinario; ciclo cerrado salvo que el malestar se repita"
        else:
            accion = "Acompañamiento verbal conversacional"
            retro = "Sin riesgo detectado; ciclo cerrado"

    respuesta_final = limpiar_texto_natural(respuesta_final)

    # RETROALIMENTACIÓN: persistencia inmediata del turno completo
    try:
        guardar_registro_interaccion(
            tipo_entrada=tipo_entrada,
            contenido=contenido,
            clasificacion=categoria,
            nivel_riesgo=nivel,
            alerta=alerta_info is not None,
            decision=decision,
            accion=accion,
            respuesta_chatbot=respuesta_final,
            retroalimentacion=retro,
            id_interaccion=id_interaccion,
        )
    except Exception:
        logger.exception("No se pudo guardar el historial de la interacción")

    return respuesta_final
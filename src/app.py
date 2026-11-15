import streamlit as st
from chatbot_logic import limpiar_texto_natural, procesar_interaccion, registrar_inicio_conversacion, precalentar_recursos

st.set_page_config(
    page_title="Asistente Personal de Apoyo",
    page_icon="💙",
    layout="centered"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-title { color: #2563EB; font-weight: bold; }
    .chat-text { font-size: 1.15rem !important; line-height: 1.6 !important; }
    </style>
""", unsafe_allow_html=True)

# Carga (una sola vez, en caché) el cliente de Gemini, el dataset y el modelo, para que el primer mensaje no espere
precalentar_recursos()

st.markdown("<h1 class='main-title'>💙 Mi Asistente de Compañía</h1>", unsafe_allow_html=True)
st.write("Un espacio tranquilo para conversar mediante voz o texto.")


st.components.v1.html("""
    <div style="margin-bottom: 10px;">
        <button onclick="parent.window.speechSynthesis.cancel(); window.speechSynthesis.cancel();" style="background: #dc2626; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px;">
            Silenciar voz 🔇
        </button>
    </div>
""", height=45)

if "historial_chat" not in st.session_state:
    st.session_state.historial_chat = [
        {"role": "assistant", "content": "Hola. ¿En qué puedo ayudarte hoy?"}
    ]

# Inicio de conversación: se registra una sola vez por sesión en historial_interacciones.json
if "conversacion_registrada" not in st.session_state:
    registrar_inicio_conversacion(st.session_state.historial_chat[0]["content"])
    st.session_state.conversacion_registrada = True


if "ultimo_audio_procesado" not in st.session_state:
    st.session_state.ultimo_audio_procesado = None

def reproducir_voz(texto):
    texto_seguro = limpiar_texto_natural(texto).replace('"', "'").replace('\n', ' ')
    script_js = f"""
    <script>
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("{texto_seguro}");
        utterance.lang = "es-MX";
        utterance.rate = 0.95; 
        window.speechSynthesis.speak(utterance);
    </script>
    """
    st.components.v1.html(script_js, height=0)


for msg in st.session_state.historial_chat:
    with st.chat_message(msg["role"]):
        texto_mostrable = limpiar_texto_natural(msg["content"])
        st.markdown(f'<p class="chat-text">{texto_mostrable}</p>', unsafe_allow_html=True)


if st.session_state.historial_chat[-1]["role"] == "assistant":
    reproducir_voz(st.session_state.historial_chat[-1]["content"])


st.markdown("### 🎙️ Dictado por Voz")
audio_grabado = st.audio_input("Presiona el botón para grabar tu mensaje de voz:")

if audio_grabado:
    audio_bytes = audio_grabado.getvalue()
    
    if audio_bytes != st.session_state.ultimo_audio_procesado:
        st.session_state.ultimo_audio_procesado = audio_bytes
        
        st.session_state.historial_chat.append({"role": "user", "content": "[Mensaje de voz enviado]"})
        with st.chat_message("user"):
            st.markdown('<p class="chat-text">🎙️ [Mensaje de voz enviado]</p>', unsafe_allow_html=True)

        # Una sola llamada a Gemini transcribe y responde; la transcripción real pasa por la Regresión Logística,
        # se evalúa el riesgo, se decide el protocolo, se alerta a enfermería si hace falta y se guarda el turno.
        with st.spinner("Escuchando y procesando audio..."):
            respuesta_bot = procesar_interaccion("audio", st.session_state.historial_chat, audio_bytes=audio_bytes)

        st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_bot})
        st.rerun()


if entrada_usuario := st.chat_input("O escribe tu mensaje aquí..."):
    st.session_state.historial_chat.append({"role": "user", "content": entrada_usuario})
    with st.chat_message("user"):
        st.markdown(f'<p class="chat-text">{entrada_usuario}</p>', unsafe_allow_html=True)

    with st.spinner("Procesando respuesta..."):
        respuesta_bot = procesar_interaccion("texto", st.session_state.historial_chat, texto=entrada_usuario)

    st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_bot})
    st.rerun()
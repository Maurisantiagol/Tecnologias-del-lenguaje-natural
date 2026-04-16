# -*- coding: utf-8 -*-
"""
chatbot.py — Motor del Restaurante Italiano
Basado en chatbot_italiano.py (pipeline end-to-end)
Carga modelos PKL pre-entrenados y expone la clase AsistenteItalianoPipeline
para ser consumida por app.py (Flask API).
"""

import joblib
import pandas as pd
import numpy as np
import re
import spacy
import random
import os

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# KEYWORDS para enrutador híbrido (compatibilidad app.py)
# ============================================================
KW_BOOK   = {'reserve', 'book', 'table', 'reservation', 'spot', 'seats', 'booking'}
KW_MENU   = {'menu', 'categories', 'options', 'what do you have', 'what do you serve'}
KW_RECIPE = {'recipe', 'cook', 'how to make', 'directions', 'instructions', 'prepare'}
KW_FOOD   = {
    'vegan', 'nut', 'gluten', 'recommend', 'want', 'eat', 'hungry',
    'pasta', 'pizza', 'salad', 'ingredients', 'risotto', 'suggest',
    'dairy', 'lactose', 'halal', 'kosher', 'plant', 'light', 'spicy',
    'tiramisu', 'gnocchi', 'ravioli', 'tortellini', 'fettuccine',
    'bruschetta', 'focaccia', 'gelato', 'antipasto', 'seafood',
    'craving', 'feel like', 'something'
}

# ============================================================
# 1. CARGA DE spaCy
# ============================================================
print("--- [chatbot.py] Cargando spaCy ---")
try:
    nlp = spacy.load("en_core_web_sm")
    stopwords_en = nlp.Defaults.stop_words
    _spacy_loaded = True
except OSError:
    print("  [WARN] spaCy model 'en_core_web_sm' no encontrado. "
          "Ejecuta: python -m spacy download en_core_web_sm")
    nlp = None
    stopwords_en = set()
    _spacy_loaded = False


def lematizar_entrada(texto: str) -> str:
    """Limpia, elimina stopwords, corrige ortografía y lematiza con spaCy (igual que chatbot_italiano.py)."""
    if not _spacy_loaded or nlp is None:
        return texto.lower().strip()
    
    texto_min = str(texto).lower()

    # Spellchecking integration
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        palabras = texto_min.split()
        corr_palabras = [spell.correction(word) or word for word in palabras]
        texto_min = " ".join(corr_palabras)
    except ImportError:
        pass

    texto_limpio = re.sub(r'[^a-zA-Z\s]', ' ', texto_min)
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    doc = nlp(texto_limpio)
    lemas_validos = [
        token.lemma_ for token in doc
        if not token.is_space
        and token.text not in stopwords_en
        and token.lemma_.isalpha()
    ]
    return " ".join(lemas_validos)


# ============================================================
# 2. SISTEMA EXPERTO — REGLAS DETERMINISTAS
# ============================================================

def validate_reservation(day_of_week: str, requested_hour: float):
    """
    Valida disponibilidad según horario oficial del restaurante.
    Lun-Jue: 08:00-22:00 | Vie-Sáb: 08:00-23:30 | Dom: 09:00-20:00
    """
    day = day_of_week.lower().strip()
    if day in ['monday', 'tuesday', 'wednesday', 'thursday']:
        if 8 <= requested_hour < 22:
            return True, "✅ Reservation confirmed! We look forward to seeing you on {day}. Mon-Thu we're open 08:00–22:00.".format(day=day.capitalize())
        else:
            return False, "❌ Sorry, Monday–Thursday we are open from 08:00 to 22:00. Please choose a time within those hours."
    elif day in ['friday', 'saturday']:
        if 8 <= requested_hour < 23.5:
            return True, "✅ Weekend reservation confirmed! We can't wait to see you on {day} — Fri & Sat we're open until 23:30! 🎉".format(day=day.capitalize())
        else:
            return False, "❌ Sorry, Friday–Saturday we are open from 08:00 to 23:30."
    elif day == 'sunday':
        if 9 <= requested_hour < 20:
            return True, "✅ Sunday reservation confirmed! See you on Sunday. We're open 09:00–20:00. 🍝"
        else:
            return False, "❌ Sorry, on Sundays we are open from 09:00 to 20:00."
    else:
        return False, "❓ Unrecognized day. Please provide a valid day of the week (e.g. Monday, Friday)."


def extraer_entidades_dieteticas(texto: str) -> dict:
    """Detecta restricciones dietéticas en el texto del usuario."""
    t = texto.lower()
    return {
        'vegan'      : ('vegan' in t or 'plant based' in t or 'plant-based' in t
                        or 'no meat' in t or 'no animal' in t),
        'gluten_free': ('gluten' in t and (
                        'free' in t or 'no ' in t or 'intolerant' in t
                        or 'allergy' in t or 'allergic' in t or 'intolerance' in t)),
        'nut_free'   : ('nut' in t and (
                        'free' in t or 'no ' in t or 'allergic' in t or 'allergy' in t)),
        'dairy_free' : (('dairy' in t or 'lactose' in t or 'milk' in t) and (
                        'free' in t or 'no ' in t or 'intolerant' in t or 'allergy' in t)) or 'dairy free' in t or 'no dairy' in t or 'no milk' in t or 'lactose intolerant' in t,
        'quick'      : ('quick' in t or 'fast' in t or 'hurry' in t
                        or 'speed' in t or 'rapid' in t or 'express' in t),
        'halal'      : 'halal' in t or 'no pork' in t or 'no alcohol' in t or 'muslim' in t,
        'kosher'     : 'kosher' in t or 'jewish dietary' in t,
    }


# ============================================================
# 3. MOTOR SEMÁNTICO — TF-IDF + Cosine Similarity
# ============================================================

def get_recommendations_api(antojo: str, dietas: dict,
                             df_base, vectorizador, matriz_completa,
                             top_n: int = 3) -> list:
    """
    Retorna recomendaciones como lista de dicts (listo para jsonify).
    Incluye dietary_badges y dietary_profile para el frontend.
    """
    vec_antojo = vectorizador.transform([antojo])
    similitudes = cosine_similarity(vec_antojo, matriz_completa).flatten()
    # Ligera aleatoriedad en empates para variar resultados
    ruido = np.random.uniform(0, 0.0001, similitudes.shape)
    indices = (similitudes + ruido).argsort()[::-1]

    results = []
    for idx in indices:
        if similitudes[idx] < 0.15 or len(results) >= top_n:
            break
        receta = df_base.iloc[idx]

        # Filtros del sistema experto
        if dietas.get('vegan')      and not bool(receta.get('is_vegan', False)):      continue
        if dietas.get('gluten_free') and not bool(receta.get('is_gluten_free', False)): continue
        if dietas.get('nut_free')   and bool(receta.get('has_nuts', False)):           continue
        if dietas.get('dairy_free') and not bool(receta.get('is_dairy_free', False)):  continue
        if dietas.get('halal')      and not bool(receta.get('is_halal', False)):       continue
        if dietas.get('kosher')     and not bool(receta.get('is_kosher', False)):      continue
        if dietas.get('quick')      and float(receta.get('est_prep_time_min', 999)) > 30: continue

        # Instrucciones limpias
        pasos_crudos = str(receta.get('directions_text', ''))
        pasos_limpios = re.sub(r'\[|\]|\\\'|\"', '', pasos_crudos)
        instrucciones = [p.strip().capitalize() for p in pasos_limpios.split(', ') if p.strip()]

        # Dietary badges para el frontend
        badges = []
        if bool(receta.get('is_vegan', False)):       badges.append({'label': 'Vegan',       'icon': '🌱'})
        if bool(receta.get('is_gluten_free', False)):  badges.append({'label': 'Gluten-Free', 'icon': '🌾'})
        if bool(receta.get('is_dairy_free', False)):   badges.append({'label': 'Dairy-Free',  'icon': '🥛'})
        if bool(receta.get('is_nut_free', False)):     badges.append({'label': 'Nut-Free',    'icon': '🥜'})
        if bool(receta.get('is_halal', False)):        badges.append({'label': 'Halal',       'icon': '☪️'})
        if bool(receta.get('is_kosher', False)):       badges.append({'label': 'Kosher',      'icon': '✡️'})

        results.append({
            'title'        : str(receta['recipe_title']),
            'ingredients'  : str(receta.get('ingredients', '')),
            'course'       : str(receta.get('course_list', '')),
            'time'         : float(receta.get('est_prep_time_min', 0) or 0),
            'directions'   : instrucciones,
            'match'        : float(similitudes[idx]),
            'dietary_badges': badges,
            'has_nuts'     : bool(receta.get('has_nuts', False)),
        })
    return results


# ============================================================
# 4. CLASE PIPELINE — CHATBOT CON ESTADO CONVERSACIONAL
# ============================================================

class AsistenteItalianoPipeline:
    """
    Chatbot multi-turno para restaurante italiano.
    Encapsula NLU (SVM), motor TF-IDF y sistema experto de reservas.
    """

    # Opciones del menú principal (precalculadas 1 vez)
    _opciones_menu_cache = None

    def __init__(self, nlu_model, tfidf_vectorizer, matriz_menu_mat, df_menu_df, df_nlp_df):
        self.nlu_model    = nlu_model
        self.vectorizador = tfidf_vectorizer
        self.matriz_menu  = matriz_menu_mat
        self.df_menu      = df_menu_df   # df_safe (con has_nuts)
        self.df_nlp       = df_nlp_df    # df_nlp  (con ingredients_norm)

        self._reset_session()

        if AsistenteItalianoPipeline._opciones_menu_cache is None:
            categorias = self.df_menu['course_list'].dropna().astype(str)
            categorias = categorias.str.replace(r'[\[\]\'\"]', '', regex=True)
            AsistenteItalianoPipeline._opciones_menu_cache = ", ".join(
                categorias.str.split(',').explode().str.strip().str.lower()
                .value_counts().head(8).index.str.title().tolist()
            )

    def _reset_session(self):
        self.session_state = {
            "intent_actual": None,
            "datos_reserva": {"dia": None, "hora": None}
        }

    def reset(self):
        """Reinicia la sesión conversacional."""
        self._reset_session()

    def procesar_mensaje(self, mensaje_usuario: str) -> dict:
        """
        Procesa un mensaje y devuelve un dict con:
          - reply (str)
          - intent (str)
          - recipes (list)
        """
        texto_limpio = mensaje_usuario.strip()
        low = texto_limpio.lower()

        # ---- ESCAPE HATCH GLOBAL ----
        escape_words = ['cancel', 'exit', 'quit', 'stop', 'no', 'nevermind']
        if any(word in low.split() for word in escape_words):
            self._reset_session()
            return {
                "reply": "No problem! Let's cancel that. What else can I help you with?",
                "intent": "Cancel",
                "recipes": []
            }

        # ---- Multi-turno: si hay reserva en curso ----
        if self.session_state["intent_actual"] == "Book_Table":
            respuesta = self._continuar_reserva(low)
            return {"reply": respuesta, "intent": "Book_Table", "recipes": []}

        # ---- Lematización para NLU ----
        texto_lem = lematizar_entrada(low)
        if not texto_lem:
            texto_lem = low

        entidades = extraer_entidades_dieteticas(low)

        # ---- Enrutador híbrido (Reglas > ML) ----
        if any(kw in low for kw in ['cancel', 'modify', 'change', 'reschedule', 'update my booking',
                                       'update reservation', 'modify reservation', "won't make it"]):
            intent = "Modify_Booking"
        elif len(set(low.split()) & KW_BOOK) > 0 or any(p in low for p in ['book', 'reserve', 'reservation', 'table for']):
            intent = "Book_Table"
        elif any(kw in low for kw in ['menu', 'categories', 'what do you have', 'what do you serve', 'what can i eat']):
            intent = "Query_Menu"
        elif any(kw in low for kw in ['does', 'is there', 'contains', 'contain', 'ingredient',
                                       'allergic', 'intolerant', 'have in', 'made with', 'put in']):
            intent = "Query_Ingredients"
        else:
            pred = self.nlu_model.predict([texto_lem])[0]
            conf = max(self.nlu_model.predict_proba([texto_lem])[0])
            intent = pred if conf >= 0.40 else "Unknown"

        # Smart Fallback
        if intent == "Unknown" and (any(entidades.values()) or len(set(low.split()) & KW_FOOD) > 0):
            intent = "Recommend_Food"

        return self._ejecutar_accion(intent, texto_limpio, texto_lem, entidades)

    def _ejecutar_accion(self, intent: str, texto_original: str,
                          texto_lem: str, entidades: dict) -> dict:
        base = {"intent": intent, "recipes": []}

        if intent == "Book_Table":
            # Intentar extracción directa de día y hora en un solo mensaje
            dias = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            dia_detectado = next((d for d in dias if d in texto_original.lower()), None)
            hora_detectada = self._parse_hora(texto_original.lower())

            if dia_detectado and hora_detectada is not None:
                ok, msg = validate_reservation(dia_detectado, hora_detectada)
                self._reset_session()
                # Personalización: respuesta menos robótica si la reserva es válida
                if ok:
                    msg = f"Perfect! Let me check availability for {dia_detectado.capitalize()} at {hora_detectada}. " + msg + "\nHow many people will be in your party?"
                return {**base, "reply": msg}
            else:
                self.session_state["intent_actual"] = "Book_Table"
                if dia_detectado:
                    self.session_state["datos_reserva"]["dia"] = dia_detectado
                    return {**base, "reply": f"Perfect! I'll get you a table for {dia_detectado.capitalize()}. At what time would you like to arrive? (e.g. 8pm, 20:00)"}
                return {**base, "reply": "Sure! I'll be happy to get you a table. What day of the week are you planning to visit? (e.g. Friday, Saturday)"}

        elif intent == "Query_Menu":
            opciones = AsistenteItalianoPipeline._opciones_menu_cache
            return {**base,
                    "reply": f"Here are some highlights from our main selection:\n**{opciones}**.\n\nWould you like me to recommend something specific? Try asking: *'Recommend me a quick vegan pasta'* or *'Something with mushrooms'*!"}

        elif intent in ("Recommend_Food", "Query_Ingredients", "Descubrir_Comida", "Discover_Food"):
            recipes = get_recommendations_api(
                texto_lem, entidades,
                self.df_nlp, self.vectorizador, self.matriz_menu
            )
            if recipes:
                reply = "🍝 Here are some great options based on your request!\n\nWould you like to know more about any of these dishes, or shall I help you make a reservation?"
            else:
                reply = "😔 I'm sorry, I couldn't find a dish that matches those exact flavors and dietary needs. Try being a bit more general!"
            return {**base, "reply": reply, "recipes": recipes}

        elif intent == "Modify_Booking":
            return {**base,
                    "reply": "📞 To modify or cancel your booking, please call our front desk directly at **+1 (555) 019-2024**. We're happy to help!\n\nWould you like me to note down your name and preferred new time so the team can reach you?"}

        else:
            fallbacks = [
                "🤔 I didn't quite catch that. You can ask me to recommend a dish, show the menu, or book a table!",
                "😅 Sorry, I'm just a restaurant bot. Could you try asking about our menu or making a reservation?",
                "🍝 I'm not sure I understand. Feel free to ask about dietary options, book a table, or browse our menu!"
            ]
            return {**base, "reply": random.choice(fallbacks)}

    def _continuar_reserva(self, texto: str) -> str:
        """Continúa el flujo multi-turno de reserva."""
        if not self.session_state["datos_reserva"]["dia"]:
            dias = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            dia_detectado = next((d for d in dias if d in texto), None)
            if dia_detectado:
                self.session_state["datos_reserva"]["dia"] = dia_detectado
                return f"📅 Got it — **{dia_detectado.capitalize()}**! And at what time? (e.g. 8pm, 20:00)"
            else:
                return "🗓️ I didn't catch the day. Please tell me the day of the week (e.g. Monday, Friday, Saturday)."

        elif self.session_state["datos_reserva"]["hora"] is None:
            hora = self._parse_hora(texto)
            if hora is not None:
                dia = self.session_state["datos_reserva"]["dia"]
                ok, msg = validate_reservation(dia, hora)
                self._reset_session()
                return msg
            else:
                return "⏰ Sorry, I didn't catch the time. Please use a format like **8pm**, **20:00**, or **14.5** for 2:30 PM."

        self._reset_session()
        return "Sorry, something went wrong. Let's start over — what day would you like to book?"

    @staticmethod
    def _parse_hora(texto: str):
        """Extrae la hora de un texto. Retorna float (24h) o None."""
        # 1. Buscar formatos de hora explícitos con regex primero
        match_explicito = re.search(r'\b(\d{1,2}):(\d{2})\s?(am|pm)?\b|\b(\d{1,2})\s?(am|pm)\b', texto)
        if match_explicito:
            if match_explicito.group(1): # Format hh:mm [am/pm]
                h = int(match_explicito.group(1))
                m = int(match_explicito.group(2))
                ampm = match_explicito.group(3)
            else: # Format h [am/pm]
                h = int(match_explicito.group(4))
                m = 0
                ampm = match_explicito.group(5)
            
            if ampm == 'pm' and h < 12:
                h += 12
            elif ampm == 'am' and h == 12:
                h = 0
            return h + (m / 60.0)
            
        # 2. Si no hay formatos explícitos, buscar números sueltos o decimales
        match_numero = re.search(r'\b(\d{1,2}(?:\.\d)?)\b', texto)
        if match_numero:
            try:
                h = float(match_numero.group(1))
                if h <= 24:
                   return h
            except ValueError:
                pass
        return None


# ============================================================
# 5. CARGA DE MODELOS PKL (SISTEMA EN FRÍO)
# ============================================================
print("--- [chatbot.py] Cargando modelos PKL ---")

# Ruta absoluta para evitar errores de ejecución desde carpetas externas
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
PATH_PLK = os.path.join(DIR_ACTUAL, "PLK")

try:
    modelo_nlu          = joblib.load(os.path.join(PATH_PLK, 'modelo_nlu_svm.pkl'))
    vectorizador_tfidf  = joblib.load(os.path.join(PATH_PLK, 'vectorizador_tfidf.pkl'))
    matriz_menu         = joblib.load(os.path.join(PATH_PLK, 'matriz_menu.pkl'))
    df_safe             = pd.read_pickle(os.path.join(PATH_PLK, 'dataset_seguro.pkl'))
    df_nlp              = pd.read_pickle(os.path.join(PATH_PLK, 'dataset_lematizado.pkl'))
    print("✅ Todos los modelos cargados correctamente.")
except FileNotFoundError as e:
    print(f"❌ Error al cargar PKL: {e}")
    print(f"   Buscando en: {PATH_PLK}")
    print("   Por favor, ejecuta primero train_models.py para generar los modelos.")
    raise

# ============================================================
# 6. INSTANCIA GLOBAL DEL ASISTENTE (usada por app.py)
# ============================================================
asistente = AsistenteItalianoPipeline(
    nlu_model        = modelo_nlu,
    tfidf_vectorizer = vectorizador_tfidf,
    matriz_menu_mat  = matriz_menu,
    df_menu_df       = df_safe,
    df_nlp_df        = df_nlp,
)
print("✅ AsistenteItalianoPipeline listo.")


# ============================================================
# 7. CICLO INTERACTIVO (ejecución directa)
# ============================================================
def run_console_chatbot():
    print("\n" + "=" * 60)
    print("  ASISTENTE VIRTUAL — RESTAURANTE ITALIANO")
    print("  (escribe 'salir' para terminar)")
    print("=" * 60)
    print("Bot: Buongiorno! 👨‍🍳 How can I help you today?\n")

    while True:
        entrada = input("You: ").strip()
        if entrada.lower() in ['salir', 'quit', 'exit', 'bye']:
            print("Bot: Arrivederci! 🍝")
            break
        resultado = asistente.procesar_mensaje(entrada)
        print(f"Bot [{resultado['intent']}]: {resultado['reply']}")
        for r in resultado.get('recipes', []):
            print(f"   -> {r['title']} (Match: {r['match']:.2f})")
        print()


if __name__ == "__main__":
    run_console_chatbot()
# -*- coding: utf-8 -*-
"""
app.py — Flask REST API para el Chatbot del Restaurante Italiano
Delega toda la lógica conversacional a chatbot.py (AsistenteItalianoPipeline).
"""

import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Inicializar Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Permite peticiones cross-origin desde el frontend

# ── Importar el módulo chatbot (carga modelos en el import) ───────────────────
try:
    import chatbot
    print("✅ chatbot.py importado correctamente.")
except Exception as e:
    print(f"❌ Error importando chatbot.py: {e}")
    traceback.print_exc()
    raise


# ==============================================================================
# POST /api/chat  — Endpoint principal de conversación
# ==============================================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Recibe: { "message": "texto del usuario" }
    Retorna:
      {
        "reply":   "texto de respuesta del bot",
        "intent":  "intención detectada",
        "recipes": [ { title, ingredients, course, time, directions,
                        dietary_badges, has_nuts, match } ]
      }
    """
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({"error": "El campo 'message' es requerido."}), 400

    user_input = data['message'].strip()
    if not user_input:
        return jsonify({"error": "El mensaje no puede estar vacío."}), 400

    try:
        resultado = chatbot.asistente.procesar_mensaje(user_input)
        return jsonify(resultado)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# POST /api/reset  — Reinicia la sesión conversacional
# ==============================================================================
@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reinicia el estado del chatbot (flujo multi-turno de reservas)."""
    try:
        chatbot.asistente.reset()
        return jsonify({"status": "ok", "message": "Sesión reiniciada."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# GET /api/suggest  — Autocompletado type-ahead
# ==============================================================================
@app.route('/api/suggest', methods=['GET'])
def suggest():
    """
    Recibe: ?q=<query>
    Retorna: { "suggestions": ["titulo1", "titulo2", ...] }
    Mezcla sugerencias de intención + títulos de recetas reales.
    """
    query = request.args.get('q', '').lower().strip()
    if not query or len(query) < 2:
        return jsonify({"suggestions": []})

    try:
        # Sugerencias por intención (frases de ejemplo)
        intent_tips = []
        if any(k in query for k in ['veg', 'plant']):
            intent_tips.append("Recommend me a vegan dish 🌱")
        if any(k in query for k in ['glu', 'wheat']):
            intent_tips.append("Gluten-free options please 🌾")
        if any(k in query for k in ['nut', 'almond', 'walnut']):
            intent_tips.append("I have a nut allergy, what can I eat?")
        if any(k in query for k in ['pas', 'spag', 'penne']):
            intent_tips.append("Recommend me a pasta dish 🍝")
        if any(k in query for k in ['piz']):
            intent_tips.append("Show me your pizza options 🍕")
        if any(k in query for k in ['res', 'book', 'tab']):
            intent_tips.append("Book a table for Friday at 8pm 📅")
        if any(k in query for k in ['ris', 'rice']):
            intent_tips.append("Recommend me a risotto 🍚")
        if any(k in query for k in ['dai', 'lact']):
            intent_tips.append("Dairy-free options please 🥛")

        # Sugerencias por título de receta
        mask = chatbot.df_safe['recipe_title'].str.lower().str.contains(
            query, na=False, regex=False
        )
        recipe_titles = chatbot.df_safe[mask]['recipe_title'].head(5).tolist()

        combined = intent_tips + recipe_titles
        # Deduplicar y limitar
        seen = set()
        unique = []
        for s in combined:
            low = s.lower()
            if low not in seen:
                seen.add(low)
                unique.append(s)
            if len(unique) >= 6:
                break

        return jsonify({"suggestions": unique})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# GET /api/health  — Health check
# ==============================================================================
@app.route('/api/health', methods=['GET'])
def health():
    """Verificación de estado del servidor."""
    return jsonify({
        "status": "ok",
        "model": "SVM + TF-IDF (chatbot_italiano pipeline)",
        "recipes_loaded": len(chatbot.df_safe),
    })


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🚀 Servidor iniciando en http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
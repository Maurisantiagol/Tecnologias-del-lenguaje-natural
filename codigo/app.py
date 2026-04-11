from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Import functions and models directly from the chatbot script.
# Since chatbot.py loads models on import, this will make the assets available immediately.
try:
    import chatbot
except Exception as e:
    print("Error importing chatbot modules:", e)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for the Vite frontend

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles conversational/query requests based on the hybrid logic."""
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Missing message field"}), 400
    
    user_input = data['message'].strip()
    low_input = user_input.lower()
    intent = "Unknown"

    import re
    # Rule-based + ML routing from chatbot.py
    if len(set(low_input.split()) & chatbot.KW_BOOK) > 0:
        intent = "Book_Table"
    elif any(kw in low_input for kw in chatbot.KW_MENU):
        intent = "Query_Menu"
    elif len(set(low_input.split()) & chatbot.KW_FOOD) > 0:
        intent = "Discover_Food"
    else:
        # ML fallback
        intent = chatbot.modelo_nlu.predict([user_input])[0]
        if max(chatbot.modelo_nlu.predict_proba([user_input])[0]) < 0.40:
            intent = "Unknown"

    response_data = {
        "intent": intent,
        "reply": "",
        "recipes": []
    }

    try:
        # Execution Branches
        if intent == "Discover_Food":
            dietas = chatbot.extraer_entidades_dieteticas(user_input)
            antojo = user_input
            
            # Use the new API-friendly recommendation function
            recipes = chatbot.get_recommendations_api(
                antojo, 
                dietas, 
                chatbot.df_nlp, 
                chatbot.vectorizador_tfidf, 
                chatbot.matriz_menu
            )
            response_data["recipes"] = recipes
            if recipes:
                response_data["reply"] = "Here is what I found based on your cravings and dietary needs. The recipe steps are included in the results!"
            else:
                response_data["reply"] = "I couldn't find a dish matching those flavors and dietary needs."

        elif intent == "Query_Menu":
            response_data["reply"] = "We have Main, Side, Bread, Soup, and Dessert categories. Try asking 'quick vegan options' or 'pasta recommendations'."

        elif intent == "Book_Table":
            # Extracción Inteligente de Día y Hora
            dias_semana = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            dia_detectado = next((d for d in dias_semana if d in low_input), None)
            
            # Buscar hora usando regex (e.g., 20, 8pm, 19:30)
            hora_match = re.search(r'(\d{1,2})(?:\s?:\s?(\d{2}))?\s?(am|pm)?', low_input)
            hora_detectada = None
            if hora_match:
                h = int(hora_match.group(1))
                m = int(hora_match.group(2)) if hora_match.group(2) else 0
                ampm = hora_match.group(3)
                if ampm == 'pm' and h < 12: h += 12
                elif ampm == 'am' and h == 12: h = 0
                hora_detectada = h + (m / 60.0)

            # Si tenemos ambos, confirmar de inmediato
            if dia_detectado and hora_detectada is not None:
                ok, msg = chatbot.validate_reservation(dia_detectado, hora_detectada)
                response_data["reply"] = msg
            else:
                pedidos = []
                if not dia_detectado: pedidos.append("day of the week")
                if hora_detectada is None: pedidos.append("time (e.g. 20:00 or 8pm)")
                response_data["reply"] = f"I'd love to help! Please tell me the {' and '.join(pedidos)} you would like to book."

        else:
            response_data["reply"] = "I'm not sure I caught that. You can ask for our menu loosely ('recommend me a quick vegan pasta') or about reservations."

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify(response_data)

@app.route('/api/suggest', methods=['GET'])
def suggest():
    """Provides type-ahead autocompletion using the df_safe titles."""
    query = request.args.get('q', '').lower()
    if not query or len(query) < 2:
        return jsonify({"suggestions": []})
        
    try:
        # Filter recipe titles that contain the query
        mask = chatbot.df_safe['recipe_title'].str.lower().str.contains(query, na=False)
        matches = chatbot.df_safe[mask]['recipe_title'].head(5).tolist()
        
        # Hardcode some query intent suggestions if it matches generic food words
        general_suggestions = []
        if 'veg' in query: general_suggestions.append("Vegan options")
        if 'glu' in query: general_suggestions.append("Gluten-free options")
        if 'pas' in query: general_suggestions.append("Recommend me a pasta")
        if 'piz' in query: general_suggestions.append("Recommend me a pizza")
            
        return jsonify({
            "suggestions": general_suggestions + matches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

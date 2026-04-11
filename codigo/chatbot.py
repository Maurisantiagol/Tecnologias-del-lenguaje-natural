import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

KW_BOOK   = {'reserve','book','table','reservation','spot','seats'}
KW_MENU   = {'menu','categories'}
KW_RECIPE = {'recipe','cook','how to make','directions','instructions','prepare'}
KW_FOOD   = {'vegan','nut','gluten','recommend','want','eat','hungry',
             'pasta','pizza','salad','ingredients','risotto','suggest',
             'dairy','lactose','halal','kosher','plant','light','spicy'}

# --- 1. LÓGICA DEL SISTEMA EXPERTO (REGLAS DETERMINISTAS) ---

def validate_reservation(day_of_week, requested_hour):
    """Valida disponibilidad según horario oficial."""
    day = day_of_week.lower()
    if day in ['monday', 'tuesday', 'wednesday', 'thursday']:
        return (True, "Confirmed!") if 8 <= requested_hour < 22 else (False, "Open 08:00-22:00")
    elif day in ['friday', 'saturday']:
        return (True, "Weekend confirmed!") if 8 <= requested_hour < 23.5 else (False, "Open 08:00-23:30")
    elif day == 'sunday':
        return (True, "Sunday confirmed!") if 9 <= requested_hour < 20 else (False, "Open 09:00-20:00")
    return False, "Invalid day."

def extraer_entidades_dieteticas(texto):
    """Detecta restricciones dietéticas en el texto."""
    texto = texto.lower()
    return {
        'vegan'      : ('vegan' in texto or 'plant based' in texto or 'plant-based' in texto),
        'gluten_free': ('gluten' in texto and ('free' in texto or 'no ' in texto or 'intolerant' in texto or 'allergy' in texto or 'allergic' in texto or 'intolerance' in texto)),
        'nut_free'   : ('nut' in texto and ('free' in texto or 'no ' in texto or 'allergic' in texto or 'allergy' in texto)),
        'dairy_free' : (('dairy' in texto or 'lactose' in texto) and ('free' in texto or 'no ' in texto or 'intolerant' in texto or 'allergy' in texto)),
        'quick'      : ('quick' in texto or 'fast' in texto or 'hurry' in texto or 'speed' in texto or 'rapid' in texto),
    }

# --- 2. MOTOR SEMÁNTICO (BÚSQUEDA MATEMÁTICA) ---

def buscador_semantico_comida(antojo, dietas, df_base, vectorizador, matriz_completa, top_n=3):
    """Cruza similitud de coseno con filtros de salud."""
    vec_antojo = vectorizador.transform([antojo])
    similitudes = cosine_similarity(vec_antojo, matriz_completa).flatten()
    indices = similitudes.argsort()[::-1]
    
    found = 0
    print("\n[Chef Bot]: Let me see...")
    for idx in indices:
        if similitudes[idx] < 0.05 or found >= top_n: break
        receta = df_base.iloc[idx]
        
        # Filtros del Sistema Experto
        if dietas['vegan'] and not receta['is_vegan']: continue
        if dietas['gluten_free'] and not receta['is_gluten_free']: continue
        if dietas['nut_free'] and receta['has_nuts']: continue
        if dietas.get('quick') and receta.get('est_prep_time_min', 999) > 30: continue
        
        print(f"\n-> {receta['recipe_title']} (Match: {similitudes[idx]:.2f})")
        print(f"   Ingredients: {receta['ingredients']}")
        found += 1
    
    if found == 0:
        print("Bot: I couldn't find a dish matching those flavors and dietary needs.")

def get_recommendations_api(antojo, dietas, df_base, vectorizador, matriz_completa, top_n=3):
    """Retorna los resultados como una lista de diccionarios en lugar de imprimir."""
    vec_antojo = vectorizador.transform([antojo])
    similitudes = cosine_similarity(vec_antojo, matriz_completa).flatten()
    # Añadimos ligera aleatoriedad a los empates para no mostrar siempre lo mismo
    empates_ruido = pd.Series(similitudes).apply(lambda x: x + np.random.uniform(0, 0.0001)).values
    indices = empates_ruido.argsort()[::-1]
    
    results = []
    for idx in indices:
        if similitudes[idx] < 0.05 or len(results) >= top_n: break
        receta = df_base.iloc[idx]
        
        # Filtros del Sistema Experto
        if dietas.get('vegan') and not bool(receta.get('is_vegan', False)): continue
        if dietas.get('gluten_free') and not bool(receta.get('is_gluten_free', False)): continue
        if dietas.get('nut_free') and bool(receta.get('has_nuts', False)): continue
        if dietas.get('dairy_free') and not bool(receta.get('is_dairy_free', False)): continue
        if dietas.get('quick') and float(receta.get('est_prep_time_min', 999)) > 30: continue
        
        # Procesar las instrucciones limpiamente
        pasos_crudos = str(receta.get('directions_text', ''))
        pasos_limpios = re.sub(r'\[|\]|\\\'|\"', '', pasos_crudos)
        instrucciones_array = [p.strip().capitalize() for p in pasos_limpios.split(', ') if p.strip()]

        results.append({
            'title': str(receta['recipe_title']),
            'ingredients': str(receta.get('ingredients', '')),
            'course': str(receta.get('course_list', '')),
            'time': float(receta.get('est_prep_time_min', 0)),
            'directions': instrucciones_array,
            'match': float(similitudes[idx])
        })
    return results

# --- 3. CARGA DE ARCHIVOS SERIALIZADOS (SISTEMA EN FRÍO) ---

print("--- LOADING SYSTEM ASSETS ---")
try:
    modelo_nlu = joblib.load('./PLK/modelo_nlu_svm.pkl')
    vectorizador_tfidf = joblib.load('./PLK/vectorizador_tfidf.pkl')
    matriz_menu = joblib.load('./PLK/matriz_menu.pkl')
    df_safe = pd.read_pickle('./PLK/dataset_seguro.pkl')
    df_nlp = pd.read_pickle('./PLK/dataset_lematizado.pkl')
    print("All assets loaded successfully!")
except FileNotFoundError:
    print("Error: PLK folder or files not found. Check your paths.")
    exit()

# --- 4. CICLO PRINCIPAL (INTERFACE) ---

def run_production_chatbot():
    print("\n--- ITALIAN RESTAURANT AI V7 (PRODUCTION READY) ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'salir']: break
        
        low_input = user_input.lower()
        intent = "Unknown"

        # Enrutador Híbrido (Reglas + ML)
        if any(w in low_input for w in ['reserve', 'book', 'table', 'reseve', 'reservation']):
            intent = "Book_Table"
        elif any(w in low_input for w in ['menu', 'categories', 'options']):
            intent = "Query_Menu"
        elif any(w in low_input for w in ['vegan', 'nut', 'gluten', 'recommend', 'want', 'eat', 'hungry', 'pasta', 'pizza']):
            intent = "Discover_Food"
        else:
            # El modelo SVM toma la decisión final si no hay palabras clave
            intent = modelo_nlu.predict([user_input])[0]
            if max(modelo_nlu.predict_proba([user_input])[0]) < 0.40:
                intent = "Unknown"

        # Ejecución
        if intent == "Book_Table":
            day = input("Bot: Which day? ").strip().lower()
            try:
                hour = float(input("Bot: At what time? (24h format): "))
                _, msg = validate_reservation(day, hour)
                print(f"Bot: {msg}")
            except: print("Bot: Invalid time.")
            
        elif intent == "Discover_Food":
            dietas = extraer_entidades_dieteticas(user_input)
            antojo = user_input
            if len(user_input.split()) <= 2:
                antojo = input("Bot: Tell me more about the flavors you want: ")
            buscador_semantico_comida(antojo, dietas, df_nlp, vectorizador_tfidf, matriz_menu)
            
        elif intent == "Query_Menu":
            print("Bot: We have Main, Side, Bread, Soup, and Dessert.")
            cat = input("Bot: What category? ").strip().lower()
            res = df_safe[df_safe['course_list'].astype(str).str.contains(cat[:-1] if cat.endswith('s') else cat, case=False)]
            if not res.empty:
                print(f"Bot: Here are 3 '{cat}' options:")
                for t in res['recipe_title'].sample(min(3, len(res))): print(f" - {t}")
            else: print(f"Bot: No results for {cat}.")
            
        else:
            print("Bot: I'm not sure. Try asking for a menu, a reservation, or a specific dish.")

if __name__ == "__main__":
    run_production_chatbot()
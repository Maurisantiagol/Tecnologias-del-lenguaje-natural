# -*- coding: utf-8 -*-
"""
train_models.py — Entrenamiento y Exportación de Modelos PKL
=============================================================
Script dedicado a:
  1. Cargar el dataset local de recetas (ya descargado en Dataset/)
  2. Filtrar y limpiar recetas italianas
  3. Corregir etiquetas dietéticas (sistema híbrido reglas + datos)
  4. Lematizar ingredientes con spaCy
  5. Construir corpus NLU (5 intenciones)
  6. Entrenar modelo SVM con GridSearchCV (mejor que el original)
  7. Entrenar vectorizador TF-IDF con parámetros optimizados
  8. Exportar todos los .pkl a Codigo/PLK/

Ejecutar:
    python train_models.py

Requiere:
    pip install scikit-learn spacy joblib pandas numpy matplotlib
    python -m spacy download en_core_web_sm
"""

import os
import re
import sys
import time
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import spacy
import kagglehub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
RUTA_BASE  = os.path.dirname(os.path.abspath(__file__))          # .../Codigo
RUTA_PLK   = os.path.join(RUTA_BASE, "PLK")
RUTA_DATA  = os.path.join(RUTA_BASE, "..", "Dataset")

# Archivos de salida
OUT_MODELO      = os.path.join(RUTA_PLK, "modelo_nlu_svm.pkl")
OUT_VECTORIZER  = os.path.join(RUTA_PLK, "vectorizador_tfidf.pkl")
OUT_MATRIZ      = os.path.join(RUTA_PLK, "matriz_menu.pkl")
OUT_SEGURO      = os.path.join(RUTA_PLK, "dataset_seguro.pkl")
OUT_LEMATIZADO  = os.path.join(RUTA_PLK, "dataset_lematizado.pkl")

os.makedirs(RUTA_PLK, exist_ok=True)

START_TIME = time.time()

def elapsed():
    return f"{time.time() - START_TIME:.1f}s"

def banner(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)

banner("TRAIN_MODELS.PY — Pipeline de Entrenamiento")
print("Destino PKL:", RUTA_PLK, flush=True)

# ============================================================
# 1. DESCARGA Y CARGA DEL DATASET
# ============================================================
banner("1/8 — Descarga y Carga del Dataset (Kaggle)")

print("  Descargando dataset de 64k platillos desde Kaggle...", flush=True)
ruta_carpeta = kagglehub.dataset_download("wafaaelhusseini/extended-recipes-dataset-64k-dishes")
archivos_csv = [f for f in os.listdir(ruta_carpeta) if f.endswith('.csv')]

if not archivos_csv:
    raise FileNotFoundError("No se encontró ningún archivo .csv en el dataset descargado.")

ruta_completa = os.path.join(ruta_carpeta, archivos_csv[0])
print(f"  Archivo descargado: {archivos_csv[0]}")
df_recetas = pd.read_csv(ruta_completa, low_memory=False)
print(f"Registros cargados: {len(df_recetas):,}  [{elapsed()}]")

# ============================================================
# 2. FILTRADO — SOLO RECETAS ITALIANAS
# ============================================================
banner("2/8 — Filtrado a Recetas Italianas")

# Columna de cocina puede variar entre datasets
col_cocina = next(
    (c for c in df_recetas.columns if 'cuisine' in c.lower()),
    None
)

if col_cocina:
    df_italiano = df_recetas[
        df_recetas[col_cocina].astype(str).str.contains('italian', case=False, na=False)
    ].copy()
else:
    print("  [WARN] No se encontró columna de cocina; usando todos los registros.")
    df_italiano = df_recetas.copy()

print(f"Platillos italianos: {len(df_italiano):,}")

COLUMNAS = [
    'recipe_title', 'course_list', 'ingredients', 'directions_text',
    'est_prep_time_min', 'difficulty', 'is_vegan', 'is_gluten_free',
    'dietary_profile', 'is_halal', 'is_kosher', 'is_nut_free', 'is_dairy_free'
]
COLUMNAS_PRESENTES = [c for c in COLUMNAS if c in df_italiano.columns]
df_menu = df_italiano[COLUMNAS_PRESENTES].copy()
df_menu = df_menu.drop_duplicates(subset=['recipe_title', 'ingredients'])
df_menu = df_menu.dropna(subset=['ingredients', 'recipe_title'])
print(f"Dimensiones del menú (sin duplicados): {df_menu.shape}  [{elapsed()}]")

# Añadir columnas faltantes con valor False
for col in ['is_vegan', 'is_gluten_free', 'is_halal', 'is_kosher', 'is_nut_free', 'is_dairy_free']:
    if col not in df_menu.columns:
        df_menu[col] = False

# ============================================================
# 3. LIMPIEZA HÍBRIDA DE ETIQUETAS DIETÉTICAS (MEJORADA)
# ============================================================
banner("3/8 — Corrección Híbrida de Etiquetas Dietéticas")

# ----- Listas de ingredientes -----
NON_VEGAN    = ["chicken","beef","pork","lamb","veal","turkey","duck","fish",
                "shrimp","crab","lobster","shellfish","egg","milk","cheese",
                "butter","cream","yogurt","honey","parmesan","feta","mozzarella",
                "ricotta","mascarpone","bacon","prosciutto","pancetta","sausage",
                "gelatin","lard","clam","mussel","anchovy","anchovies"]

DAIRY        = ["milk","cheese","butter","cream","yogurt","whey","parmesan","feta",
                "mozzarella","ricotta","mascarpone","gorgonzola","pecorino","provolone",
                "burrata","brie","casein","lactose"]

GLUTEN       = ["wheat","flour","bread","panko","soy sauce","pasta","noodles",
                "lasagna","dough","barley","rye","pizza","breadcrumbs","cracker",
                "biscuit","semolina","couscous","spelt"]

PORK_DERIV   = ["pork","bacon","ham","prosciutto","pancetta","sausage","guanciale","lard"]

ALCOHOL      = ["wine","beer","vodka","rum","whiskey","tequila","liquor","brandy",
                "grappa","limoncello","amaretto","marsala","chianti"]

SHELLFISH    = ["shrimp","crab","lobster","shellfish","clam","mussel","oyster",
                "scallop","squid","octopus","calamari","anchovy","anchovies"]

MEAT_GEN     = ["chicken","beef","lamb","veal","turkey","duck","pork"]

NUTS         = ["almond","walnut","cashew","pecan","pistachio","hazelnut","macadamia",
                "pine nut","pignoli","peanut","chestnut","praline","nutella","marzipan"]

RISKY_NUTS   = ["pesto","granola","satay","romesco","praline","baklava","marzipan","nutella"]


def contains_word(text: str, words: list) -> bool:
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')[s]?\b'
    return bool(re.search(pattern, str(text).lower()))


def analizar_receta(row) -> pd.Series:
    text = str(row['ingredients']).lower()

    # Gluten: acepta frases explícitas de libre-gluten
    if "gluten-free" in text or "gluten free" in text or "spaghetti squash" in text:
        gluten_flag = True
    else:
        gluten_flag = not contains_word(text, GLUTEN)

    has_meat  = contains_word(text, MEAT_GEN)
    has_dairy = contains_word(text, DAIRY)

    return pd.Series({
        "is_vegan_det"      : not contains_word(text, NON_VEGAN),
        "is_gluten_free_det": gluten_flag,
        "is_dairy_free_det" : not has_dairy,
        "is_halal_det"      : not contains_word(text, PORK_DERIV + ALCOHOL),
        "is_kosher_det"     : not (
            contains_word(text, PORK_DERIV)  or
            contains_word(text, SHELLFISH)   or
            (has_meat and has_dairy)
        ),
        "is_nut_free_det"   : not contains_word(text, NUTS),
        "has_nuts"          : contains_word(text, NUTS),
        "nut_risk"          : (
            "HIGH"   if contains_word(text, NUTS)
            else "MEDIUM" if contains_word(text, RISKY_NUTS)
            else "LOW"
        )
    })

print("  Analizando ingredientes...", flush=True)
results = df_menu.apply(analizar_receta, axis=1)
df_clean = pd.concat([df_menu, results], axis=1)
print(f"  Análisis completado  [{elapsed()}]")

# Correcciones
cambios = {k: 0 for k in ["vegan","gluten","dairy","halal","kosher","nuts"]}

def corregir(row):
    if row.get('is_vegan', False)      and not row['is_vegan_det']:
        row['is_vegan']=False;           cambios["vegan"]  += 1
    if row.get('is_gluten_free', False) and not row['is_gluten_free_det']:
        row['is_gluten_free']=False;     cambios["gluten"] += 1
    if row.get('is_dairy_free', False)  and not row['is_dairy_free_det']:
        row['is_dairy_free']=False;      cambios["dairy"]  += 1
    if row.get('is_halal', False)       and not row['is_halal_det']:
        row['is_halal']=False;           cambios["halal"]  += 1
    if row.get('is_kosher', False)      and not row['is_kosher_det']:
        row['is_kosher']=False;          cambios["kosher"] += 1
    if row.get('is_nut_free', False)    and not row['is_nut_free_det']:
        row['is_nut_free']=False;        cambios["nuts"]   += 1
    return row

df_clean = df_clean.apply(corregir, axis=1)

total = len(df_clean)
print("Correcciones aplicadas:")
for k, v in cambios.items():
    print(f"  {k}: {v} ({v/total*100:.2f}%)")

# Dataset seguro (sin alto riesgo de alérgenos)
df_safe = df_clean[
    (df_clean['nut_risk'] == "LOW") &
    (df_clean['has_nuts'] == False)
].copy()
print(f"Dataset seguro: {df_safe.shape}  [{elapsed()}]")

# ============================================================
# 4. PROCESAMIENTO NLP CON SPACY
# ============================================================
banner("4/8 — Lematización con spaCy")

print("  Cargando modelo spaCy...", flush=True)
try:
    nlp = spacy.load("en_core_web_sm")
    stopwords_en = nlp.Defaults.stop_words
    print("  ✅ spaCy cargado")
except OSError:
    print("  ❌ spaCy 'en_core_web_sm' no encontrado.")
    print("     Ejecuta: python -m spacy download en_core_web_sm")
    sys.exit(1)

BASURA_CULINARIA = {
    'cup','tablespoon','teaspoon','ounce','pound','chop','slice','fresh',
    'taste','lb','oz','g','ml','clove','pinch','large','small','medium',
    'cut','diced','dice','peel','grate','package','thinly','finely',
    'roughly','divided','optional','needed'
}


def normalizar_texto_nlp(texto: str) -> str:
    texto_limpio = re.sub(r'[^a-zA-Z\s]', ' ', str(texto).lower())
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    doc = nlp(texto_limpio)
    lemas = [
        token.lemma_
        for token in doc
        if not token.is_space
        and token.text not in stopwords_en
        and token.lemma_.isalpha()
        and token.lemma_ not in BASURA_CULINARIA
        and len(token.lemma_) > 2
    ]
    return " ".join(lemas)


print("  Lematizando ingredientes (puede tardar)...", flush=True)
df_nlp = df_safe.copy()
df_nlp['ingredients_norm'] = df_nlp['ingredients'].apply(normalizar_texto_nlp)
print(f"  Lematizado: {len(df_nlp):,} recetas  [{elapsed()}]")

# ============================================================
# 5. CONSTRUCCIÓN DEL CORPUS NLU — 5 INTENCIONES
# ============================================================
banner("5/8 — Corpus NLU (5 intenciones)")


def gen(intent, listas):
    """Genera combinaciones cartesianas de frases."""
    combos = list(itertools.product(*listas))
    frases = [" ".join(palabras).strip() for palabras in combos]
    return pd.DataFrame({'texto_usuario': frases, 'intent': intent})


# ---------- Book_Table ----------
book_a = gen("Book_Table", [
    ["I want to","Can I","I need to","I'd like to","Please","Help me",
     "I'm looking to","We'd like to","Is it possible to","Could we",
     "I was hoping to","We need to","I would love to","Could I possibly"],
    ["book","reserve","get","secure","arrange","make","set up","schedule"],
    ["a table","a reservation","a spot","some seats","seating",
     "a booking","a table for two","a table for four","a private table",
     "a table outside","an outdoor table","a table by the window",
     "a quiet table","a booth"],
    ["for tonight","for tomorrow","this friday","at 8 PM","for 2 people",
     "for my family","for this weekend","for saturday night","for sunday brunch",
     "for next week","for our anniversary","for a birthday dinner",
     "for a date night","for a business dinner","for 3 people","for 4 people",
     "for 6 people","for a group of 8","at 7 PM","at 9 PM",
     "at 6:30 PM","for lunch tomorrow","for dinner on friday",
     "for a special occasion","for valentines day"]
])
book_b = gen("Book_Table", [
    ["Do you have","Are there any","Is there","Can you check if there are"],
    ["available tables","open reservations","spots available","free tables","openings"],
    ["tonight","tomorrow night","this saturday","this sunday","next friday",
     "for the weekend","for two","for a party of four","at 8","after 7 PM"]
])
extra_book = pd.DataFrame({'intent': 'Book_Table', 'texto_usuario': [
    "table for two please","i need a table","book me in for 7pm",
    "can we get a spot this saturday","reserve a table for my anniversary",
    "we are 4 people need a table","i need a reservation asap",
    "can i get a table for lunch tomorrow","booking for 2 at 8",
    "do you have availability tonight","id like to come in friday evening",
    "reservation for my wife and i","can i reserve a spot for saturday lunch",
    "we are celebrating a birthday","need a table for a family of six",
    "can i book for new years eve","reserving a table for valentines",
    "i want to bring my parents for dinner","booking for business dinner wednesday",
    "private table for a proposal","outdoor seating available this weekend",
    "table by the window please","quiet corner table for two",
    "we need a high chair for a baby","we are a large group of 10",
    "do you take walk ins","is reservation required",
    "do i need to book in advance","last minute reservation possible",
    "i need a table right now","are you open tonight",
    "can we come in at 6","tonight for 2","dinner for four this friday",
    "lunch booking for tomorrow","brunch reservation sunday",
    "table for one please","solo reservation for tonight",
    "i want to dine tonight","can i come for dinner",
    "available tonight for dinner","need to make a booking",
    "could i get a reservation please","book a table for saturday",
    "grab a table for two", "resevation for 4", "save me a spot",
    "i wanna eat there tonight", "need a place to sit at 8", 
    "got an empty table for us?", "can you fit us in?", 
    "gimme a table please", "looking to get some food locally", 
    "need somewhere to dine tomorrow", "wanna secure some seats", 
    "any room for 3 guys tonight?", "need to book it rn", 
    "hold a place for me", "is there space left?", 
    "i gotta book my birthday dinner here"
]})

# ---------- Query_Menu ----------
menu_a = gen("Query_Menu", [
    ["What do you have","Show me","I want to see","Can I see","Do you have",
     "Tell me about","Can you show me","Let me look at","I'd like to browse",
     "What's available","Can you list","Give me a look at","What are your options",
     "I'm curious about","What are you serving"],
    ["for dessert","for dinner","for breakfast","for lunch","on the menu",
     "the menu","pasta options","pizza options","vegan options",
     "gluten free options","today's specials","the appetizers","the starters",
     "the main courses","the chef's specials","seafood dishes",
     "vegetarian options","the drinks menu","the wine list",
     "light dishes","soups","salads","risotto options","meat dishes",
     "fish dishes","nut-free options","dairy-free options",
     "the dessert menu","something light","the full menu"]
])
menu_b = gen("Query_Menu", [
    ["What kind of","What types of","Which","How many kinds of"],
    ["pasta","pizza","risotto","antipasto","desserts","salads","soups",
     "fish dishes","meat dishes","vegetarian dishes","vegan dishes"],
    ["do you serve","do you offer","do you have","are on the menu",
     "are available today","can I order","does the restaurant have"]
])
extra_menu = pd.DataFrame({'intent': 'Query_Menu', 'texto_usuario': [
    "whats on the menu","what can i eat here","do you have a dessert menu",
    "what pasta dishes do you have","tell me your specials",
    "what are your main courses","do you serve breakfast",
    "what are todays specials","do you have a seasonal menu",
    "is there a prix fixe menu","whats the tasting menu like",
    "do you have a kids menu","can i see the full menu",
    "what wines do you carry","what cocktails do you serve",
    "show me the starters","what soups do you have",
    "any salads on the menu","what are your side dishes",
    "do you do takeaway","can i order delivery",
    "do you have a lunch menu","what are your prices like",
    "do you change the menu often","is there a vegetarian section",
    "show me vegan dishes only","gluten free menu please",
    "dairy free options please","halal certified dishes",
    "kosher food available","what are your most popular dishes",
    "chef recommendations","staff favorites",
    "whats good here","what do you specialize in",
    "regional specialties please","traditional dishes only",
    "what is this restaurant known for","any new dishes this week",
    "what u guys serve", "show me ur pasta assortments", 
    "got any list of foods?", "lemme look at the dishes",
    "wanna see the food list", "tell me what i can order", 
    "do you guys make wood-fired pizza", "let me peek at the catalogue",
    "what are the eating options?", "read me the specials",
    "what drinks u got", "any vegetarian specialties",
    "anything good today?", "u got food for kids?", 
    "wanna know the prices", "display the meal selection"
]})

# ---------- Query_Ingredients ----------
ing_a = gen("Query_Ingredients", [
    ["Does this have","Is there","Do you put","Are there","Does the recipe include",
     "Does this dish contain","Is this made with","Does it come with"],
    ["cheese","meat","pork","nuts","cream","alcohol","gluten","seafood",
     "dairy","eggs","butter","shellfish","garlic","onion","tomato",
     "anchovies","olives","capers","mushrooms","truffle","chili",
     "pine nuts","walnuts","almonds","heavy cream","parmesan","mozzarella",
     "ricotta","mascarpone","pancetta","prosciutto","wine","beef broth"],
    ["in the pasta?","in the pizza?","in this dish?","in the dessert?",
     "in the risotto?","in the sauce?","in the filling?","in the dough?","in the broth?",""]
])
ing_b = gen("Query_Ingredients", [
    ["I am allergic to","I can't eat","I am intolerant to","I avoid",
     "I don't eat","I need to avoid","I have an allergy to",
     "I have a sensitivity to","Please no","I stay away from",
     "I cannot have","I must avoid"],
    ["nuts","dairy","gluten","cheese","seafood","pork","shellfish",
     "eggs","alcohol","soy","wheat","lactose","beef","fish",
     "tree nuts","peanuts","sesame","anchovies","onion","garlic",
     "mushrooms","cream","butter","parmesan","mozzarella","prosciutto"]
])
ing_c = gen("Query_Ingredients", [
    ["Is the","Is your"],
    ["pasta","pizza","risotto","sauce","tiramisu","carbonara",
     "lasagna","arancini","panna cotta","gelato","bruschetta",
     "focaccia","gnocchi","ravioli","tortellini","fettuccine"],
    ["vegan?","gluten free?","dairy free?","nut free?","halal?",
     "kosher?","vegetarian?","safe for lactose intolerance?",
     "suitable for nut allergies?","made without meat?",
     "made without eggs?","alcohol free?"]
])
extra_ing = pd.DataFrame({'intent': 'Query_Ingredients', 'texto_usuario': [
    "does the carbonara have cream","is there pork in this",
    "does the tiramisu have alcohol","is the pesto nut free",
    "does the lasagna have gluten","any eggs in the pasta dough",
    "is there dairy in this pasta","what is in the bolognese sauce",
    "does the arrabbiata have anchovies","is the cacio e pepe vegetarian",
    "does the amatriciana have pork","what cheese is on the pizza",
    "is the dough vegan","does the risotto use chicken stock",
    "is the broth meat based","does this contain soy",
    "is there wheat in this","does the gelato have dairy",
    "is the sorbet vegan","what is in the tiramisu",
    "does panna cotta have gelatin","is the focaccia vegan",
    "does the bruschetta have meat","any hidden animal products",
    "is the pasta egg free","does the gnocchi have eggs",
    "is the sauce cooked with wine","is there alcohol in any of the sauces",
    "does the dessert have rum","do you use MSG",
    "does this have preservatives","is the prosciutto gluten free",
    "does the pizza contain milk","is there sesame in any dish",
    "any coconut in the desserts","does this recipe include lard",
    "does the soup have cream","is the minestrone vegan",
    "what meat is in the bolognese","is the sausage pork or beef",
    "has it got dairy", "im allergic to milk", "any hidden pork in there",
    "tell me the ingredients inside", "what is this made out of?",
    "does it got flour?", "im lactose intolerant", "is it really vegan tho?",
    "sure it has no eggs?", "i cant eat peanuts", "got any tree nut in it?",
    "glten free?", "is that vegetarian safe?", "does it use real butter?",
    "can my jewish friend eat it?", "is the meat halal certified?",
    "any wheat traces?"
]})

# ---------- Recommend_Food ----------
rec_a = gen("Recommend_Food", [
    ["What do you","Can you","Please","Could you","I'd love if you could",
     "Would you mind","Do you mind","I need you to","I was hoping you could"],
    ["recommend","suggest","offer","propose","tell me about","describe"],
    ["for dinner","for breakfast","for dessert","to eat",
     "from the chef","for a date night","for a light meal",
     "for something romantic","for a group","for a birthday",
     "something hearty","something light","something spicy",
     "something creamy","something vegetarian","something vegan",
     "something gluten free","something with truffles",
     "something with mushrooms","something with seafood",
     "a pasta dish","a pizza","a risotto","an antipasto",
     "something quick to eat","a traditional Italian dish",
     "something without meat","something without dairy"]
])
rec_b = gen("Recommend_Food", [
    ["I feel like eating","I'm craving","I want to try","I'd love to have",
     "I'm in the mood for","Tonight I want","I fancy","My heart wants",
     "I really want","I keep thinking about","I'm really into"],
    ["something Italian","pasta","pizza","risotto","something cheesy",
     "something with garlic","something hearty","a light dish",
     "something traditional","comfort food","something elegant",
     "something with mushrooms","something with seafood",
     "something vegan","something gluten free","something spicy",
     "something creamy","something with truffles","a classic recipe",
     "a vegetarian option","something simple","something indulgent",
     "a regional specialty","something fresh","a warm dish",
     "a salad","a soup","something low carb","something high protein"]
])
extra_rec = pd.DataFrame({'intent': 'Recommend_Food', 'texto_usuario': [
    "i dont know what to eat","something light please","not too heavy food",
    "what do you suggest","im hungry","give me something good",
    "i want pasta but no cheese","something without meat","id like something simple",
    "i want something spicy","give me something with garlic","i feel like eating pasta",
    "im starving","i need vegan options","anything gluten free on the menu",
    "i have a gluten intolerance","i am lactose intolerant","no dairy please",
    "i avoid nuts do you have options","i follow a plant based diet",
    "vegan pasta with tomato","something light and gluten free","i want risotto",
    "do you have something with mushrooms","surprise me with something italian",
    "i prefer vegetarian food","anything creamy and cheesy",
    "i want a light appetizer","whats good today","i could go for pizza",
    "i only eat plant based","strictly vegan please","no animal products for me",
    "pescatarian options","i eat fish but not meat","low calorie options",
    "something high protein","low carb options please","keto friendly dishes",
    "no tomatoes please","allergy to shellfish","halal food only",
    "kosher options","i keep kosher","its my birthday what do you recommend",
    "anniversary dinner suggestions","romantic dinner ideas",
    "something impressive for a date","whats the most popular dish here",
    "what do most people order","your best seller please",
    "what would the chef recommend","whats the house specialty",
    "signature dish please","something savory and rich","i love umami flavors",
    "something with a lot of flavor","bold flavors please",
    "i want something mild","not too spicy please","something smoky",
    "earthy flavors like truffle or mushroom","something tangy",
    "i really love truffle","anything with black truffle",
    "i want a dish with porcini mushrooms","something with sun dried tomatoes",
    "dishes with artichoke","i love eggplant parmigiana",
    "got anything with burrata","i want fresh mozzarella",
    "a dish featuring ricotta","something with clams","i love calamari",
    "shrimp pasta please","a hearty soup","a warm broth based dish",
    "antipasto selection","i want a salad but make it interesting",
    "something to share with the table","i want a full three course meal",
    "pasta please","any pizza","risotto","just a salad","soup","antipasto",
    "tiramisu","panna cotta","gelato","bruschetta","focaccia","gnocchi",
    "ravioli","tortellini","fettuccine alfredo","spaghetti bolognese",
    "cacio e pepe","amatriciana","arrabbiata","carbonara without cream",
    "just surprise me", "i'm starving", "wanna eat something hot",
    "feed me something healthy", "what should i order?", "reccomend me something",
    "hook me up with the best dish", "kinda want some seafood today",
    "any good pasta tonight?", "give me ur tastiest option", "i need comfort food right now",
    "cant decide help me choose", "whats ur favorite item?", "find me a good vegan plate",
    "i gotta eat some dessert", "i crave strong flavors"
]})

# ---------- Modify_Booking ----------
mod_a = gen("Modify_Booking", [
    ["I want to","I need to","Can I","Please","I have to",
     "I'd like to","I must","Is it possible to","Help me",
     "Could you help me","I am looking to","I would like to",
     "Could I","Is there a way to"],
    ["cancel","change","modify","update","delete","reschedule",
     "push back","move","postpone","confirm","verify","check"],
    ["my reservation","my table","my booking","my appointment",
     "my dinner reservation","my lunch booking","our table reservation",
     "the reservation I made","a booking I placed earlier",
     "our reservation for tonight","my table for two"],
    ["for tonight","for tomorrow","for this saturday","for next week",
     "for our anniversary dinner","made last week",""]
])
mod_b = gen("Modify_Booking", [
    ["I won't be able to make it","We can't come","Something came up",
     "Our plans changed","I need to move my booking"],
    ["tonight","this friday","tomorrow night","this weekend",
     "on saturday","on sunday","this evening"]
])
extra_mod = pd.DataFrame({'intent': 'Modify_Booking', 'texto_usuario': [
    "i need to cancel my booking","can i change the time of my reservation",
    "please cancel my reservation","i have to reschedule my booking",
    "can i update my reservation","i want to cancel for tonight",
    "we cant make it anymore","something came up can i reschedule",
    "change my booking to tomorrow","move my reservation to next week",
    "i need to add a person to my booking","we went from 2 to 4 people can you update",
    "reduce my reservation to 2 guests","can i change the date of my booking",
    "is it possible to change the time","id like to change from 7pm to 9pm",
    "can i make it earlier","move my booking to friday instead",
    "i made a mistake in my reservation","i put the wrong time when i booked",
    "wrong number of people on my reservation",
    "can i get a confirmation of my booking","how do i confirm my reservation",
    "i want to check my booking details","when is my reservation",
    "can you look up my reservation","i want to add a special request",
    "can i request a specific table","i want to be seated outside",
    "can i add a birthday cake request","i need to note a food allergy",
    "please add that i am vegan to my booking","note gluten allergy please",
    "i need to cancel last minute sorry","emergency cancellation needed",
    "i am running 20 minutes late","will you hold my table if im late",
    "i may be a bit late for my reservation","how long do you hold tables",
    "i wont make it tonight i am sorry","we need to cancel our anniversary dinner",
    "plans changed can we rebook for another night",
    "i already have a reservation but want to change it",
    "delete my reservation please","remove my booking",
    "cancel it", "can we postpone to 9", "running late dont cancel",
    "i gotta skip tonight sry", "change it to thursday", "nvm im not coming",
    "wanna shift the time", "pls update my res", "gotta erase that booking",
    "make it next saturday instead", "can u alter my arrival time?",
    "we are gonna be delayed", "delay my spot by an hour", "drop the reservation",
    "i need to correct the person count", "we are 5 now instead of 3",
    "forget it, im cancelling it"
]})

# ----- Concatenación final -----
df_nlu = pd.concat(
    [book_a, book_b, extra_book,
     menu_a, menu_b, extra_menu,
     ing_a, ing_b, ing_c, extra_ing,
     rec_a, rec_b, extra_rec,
     mod_a, mod_b, extra_mod],
    ignore_index=True
)
df_nlu['texto_usuario'] = (
    df_nlu['texto_usuario']
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
    .str.lower()
)
df_nlu = df_nlu.drop_duplicates(subset=['texto_usuario']).reset_index(drop=True)

print(f"Dataset NLU total: {len(df_nlu):,} frases")
print(df_nlu['intent'].value_counts().to_string())

# ============================================================
# 6. ENTRENAMIENTO DEL MODELO SVM (CON GRID SEARCH)
# ============================================================
banner("6/8 — Entrenamiento SVM con GridSearchCV")

# Downsample para evitar sesgo
df_book = df_nlu[df_nlu['intent'] == 'Book_Table']
if len(df_book) > 5000:
    df_nlu = pd.concat([
        df_book.sample(n=5000, random_state=42),
        df_nlu[df_nlu['intent'] != 'Book_Table']
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset NLU después del Downsampling: {len(df_nlu):,} frases")
    print(df_nlu['intent'].value_counts().to_string())

X = df_nlu['texto_usuario']
y = df_nlu['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Pipeline SVM
pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ('svc', SVC(kernel='linear', probability=True, class_weight='balanced'))
])

# Grid de hiperparámetros - Reducido para acelerar el entrenamiento (manteniendo la calidad lo más alta posible)
param_grid = {
    'tfidf__max_features': [25000, None],
    'tfidf__ngram_range' : [(1, 2)],
    'svc__C'             : [1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("  Iniciando GridSearchCV  (puede tardar ~2-5 min)...", flush=True)
grid = GridSearchCV(
    pipeline_svm,
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    refit=True
)
grid.fit(X_train, y_train)

print(f"\n  Mejores parámetros: {grid.best_params_}")
print(f"  Mejor F1 (CV):      {grid.best_score_:.4f}")
modelo_nlu = grid.best_estimator_

# Evaluación en test
y_pred = modelo_nlu.predict(X_test)
f1_test = f1_score(y_test, y_pred, average='macro')
print(f"\n  F1 macro en Test: {f1_test:.4f}  [{elapsed()}]")
print("\n--- REPORTE COMPLETO ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Prueba en vivo
print("--- PRUEBA EN VIVO ---")
FRASES_PRUEBA = [
    "Hey, I need a table for 5 people this friday at 8pm",
    "Does the carbonara have any cream or parmesan cheese?",
    "Can you recommend a good vegan dessert?",
    "Show me the pizza options on the menu",
    "Cancel my reservation for tonight please",
    "I'm craving something with mushrooms and truffle",
    "I have a gluten intolerance, what can I eat?",
    "I am allergic to nuts, are there safe options?",
]
for frase in FRASES_PRUEBA:
    pred = modelo_nlu.predict([frase])[0]
    conf = max(modelo_nlu.predict_proba([frase])[0])
    estado = "✅" if conf >= 0.6 else "⚠️"
    print(f"  {estado} [{pred:20s}] ({conf:.2f})  '{frase}'")

# Matriz de confusión
labels_order = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=30)
ax.set_title("Matriz de Confusión — Modelo NLU SVM")
plt.tight_layout()
cm_path = os.path.join(RUTA_PLK, "confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"\n  Matriz de confusión guardada en: {cm_path}")

# ============================================================
# 7. MOTOR DE RECOMENDACIÓN TF-IDF (MEJORADO)
# ============================================================
banner("7/8 — Motor TF-IDF de Recomendación")

BASURA_VECTORIZER = (
    list(BASURA_CULINARIA) +
    ['cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', 'sliced',
     'chopped', 'minced', 'peeled', 'drained', 'softened', 'melted',
     'room', 'temperature', 'divided', 'package', 'about', 'use']
)

vectorizador_ingredientes = TfidfVectorizer(
    stop_words=BASURA_VECTORIZER,
    ngram_range=(1, 2),           # bigramas para capturar "pine nut", "olive oil"
    min_df=3,                     # ignora términos muy raros
    max_df=0.85,                  # ignora términos casi universales
    sublinear_tf=True
)

print("  Vectorizando ingredientes lematizados...", flush=True)
df_nlp['corpus_tfidf'] = df_nlp['recipe_title'].astype(str) + " " + df_nlp['course_list'].astype(str) + " " + df_nlp['ingredients_norm'].astype(str)
matriz_menu = vectorizador_ingredientes.fit_transform(df_nlp['corpus_tfidf'])
print(f"  Matriz: {matriz_menu.shape}  (recetas × términos)  [{elapsed()}]")
print(f"  Vocabulario: {len(vectorizador_ingredientes.vocabulary_):,} términos")

# Test rápido de similitud
test_queries = [
    "mushroom truffle risotto",
    "vegan tomato pasta",
    "seafood shrimp garlic",
]
print("\n  --- Test de Recomendación ---")
for q in test_queries:
    vec_q = vectorizador_ingredientes.transform([q])
    sims  = cosine_similarity(vec_q, matriz_menu).flatten()
    top_i = sims.argsort()[::-1][:3]
    print(f"\n  Query: '{q}'")
    for i in top_i:
        if sims[i] > 0.05:
            print(f"    [{sims[i]:.3f}] {df_nlp.iloc[i]['recipe_title']}")

# ============================================================
# 8. EXPORTACIÓN DE MODELOS PKL
# ============================================================
banner("8/8 — Exportando PKL a Codigo/PLK/")

joblib.dump(modelo_nlu,               OUT_MODELO,     compress=3)
joblib.dump(vectorizador_ingredientes, OUT_VECTORIZER, compress=3)
joblib.dump(matriz_menu,               OUT_MATRIZ,     compress=3)
df_safe.to_pickle(OUT_SEGURO)
df_nlp.to_pickle(OUT_LEMATIZADO)

print("\nArchivos generados en PLK/:")
archivos = [OUT_MODELO, OUT_VECTORIZER, OUT_MATRIZ, OUT_SEGURO, OUT_LEMATIZADO]
for f in archivos:
    size_kb = os.path.getsize(f) / 1024
    print(f"  ✅ {os.path.basename(f):35s} {size_kb:>8.1f} KB")

total_time = time.time() - START_TIME
banner(f"ENTRENAMIENTO COMPLETO — {total_time:.1f}s totales")
print(f"  F1 macro test:  {f1_test:.4f}")
print(f"  Recetas seguras: {len(df_safe):,}")
print(f"  Vocaburario TF-IDF: {len(vectorizador_ingredientes.vocabulary_):,} términos")
print(f"\n  Los modelos están listos en: {RUTA_PLK}")
print(f"  Ahora puedes ejecutar: python chatbot.py  o  python app.py\n")

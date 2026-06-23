# Chatbot para Restaurante Italiano 🍝

Este proyecto es un chatbot de extremo a extremo diseñado para un restaurante italiano. Permite a los usuarios interactuar de forma natural para realizar reservaciones, consultar el menú, preguntar por ingredientes y alérgenos, y recibir recomendaciones de platillos utilizando Procesamiento de Lenguaje Natural (NLP).

## 🌟 Características Principales

*   **Reconocimiento de Intenciones (NLU):** Utiliza un modelo SVM (Support Vector Machine) entrenado con TF-IDF para detectar intenciones como `Book_Table` (Reservar Mesa), `Query_Menu` (Consultar Menú), `Recommend_Food` (Recomendar Comida), `Query_Ingredients` (Consultar Ingredientes) y `Modify_Booking` (Modificar Reserva).
*   **Motor de Recomendación:** Recomienda platillos del menú basados en similitud del texto (Cosine Similarity) usando TF-IDF. Toma en cuenta restricciones dietéticas como dietas veganas, sin gluten, o alergias a nueces.
*   **Soporte Bilingüe (Middleware de Traducción):** Aunque el modelo NLU y el dataset operan internamente en inglés, el sistema incluye un middleware de traducción automática bidireccional (usando `deep-translator`). Esto permite a los usuarios interactuar completamente en **español**.
*   **API REST:** El backend está construido en Flask (`app.py`), exponiendo endpoints listos para ser consumidos por cualquier interfaz gráfica.
*   **Frontend Integrado:** Interfaz web lista para usarse construida con HTML, CSS y Vanilla JS.

## 🏗️ Arquitectura del Proyecto

*   `app.py`: Archivo principal que levanta el servidor Flask. Contiene los endpoints `/api/chat`, `/api/suggest` y `/api/reset`. Aquí reside la lógica del middleware de traducción Español <-> Inglés.
*   `chatbot.py` / `chatbot_italiano.py`: Núcleo de la lógica del asistente. Contiene el pipeline de procesamiento de texto con `spaCy`, la integración de los modelos de Machine Learning y la lógica de validación de reservas.
*   `frontend/`: Directorio que contiene la interfaz web (`index.html`, `app.js`).
*   `test_ruido_traduccion.py`: Script para probar la robustez del modelo de traducción y detección de intenciones inyectando "ruido" (errores tipográficos) simulando a un usuario real.
*   Archivos `.pkl`: Modelos entrenados (SVM, TF-IDF, datasets serializados) listos para producción.

## 🚀 Cómo Ejecutar el Proyecto

### 1. Requisitos Previos

Asegúrate de tener Python 3.8 o superior instalado en tu sistema.
Clona este repositorio o descarga el código fuente y navega hasta el directorio del proyecto en tu terminal.

### 2. Instalación de Dependencias

Se recomienda crear un entorno virtual para no afectar tus dependencias globales:

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate
# Activar entorno (Mac/Linux)
source venv/bin/activate
```

Instala las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt
```

*(Nota: el archivo `requirements.txt` ya incluye el modelo de lenguaje de spaCy `en_core_web_sm`).*

### 3. Ejecutar el Servidor Backend (Flask)

Para levantar la API, simplemente ejecuta:

```bash
python app.py
```

Deberías ver un mensaje en consola indicando que el servidor se está ejecutando en `http://0.0.0.0:5000` (o `http://127.0.0.1:5000`).

### 4. Ejecutar la Interfaz de Usuario (Frontend)

El frontend no requiere de un servidor web adicional (como Node.js o Nginx) para pruebas básicas.
Simplemente **abre el archivo `frontend/index.html` directamente en tu navegador web**.

*Asegúrate de que el backend de Flask esté corriendo en el puerto 5000, ya que el archivo `app.js` hace las peticiones apuntando a `http://localhost:5000/api/chat`.*

## 🌐 ¿Cómo funciona el Middleware de Traducción?

El motor principal fue entrenado en inglés. Para ofrecer una experiencia en español, el endpoint `/api/chat` en `app.py` hace lo siguiente:

1.  Recibe el texto en español junto a una bandera `translate_es: true`.
2.  Traduce el texto de **Español a Inglés** utilizando `GoogleTranslator`.
3.  El texto en inglés es procesado por el modelo SVM y las reglas del chatbot.
4.  El chatbot genera una respuesta y/o una lista de recomendaciones (todo en inglés).
5.  El servidor intercepta esa respuesta y traduce el texto principal (`reply`), los títulos de las recetas (`title`) y las instrucciones (`directions`) de **Inglés a Español** antes de enviar el JSON de vuelta al cliente.

## 🧪 Pruebas de Robustez (Ruido)

Puedes probar cómo reacciona el sistema ante faltas de ortografía (muy útiles para medir su tolerancia a errores reales de usuarios):

```bash
python test_ruido_traduccion.py
```

Esto generará automáticamente casos de prueba inyectando ruido en las frases y exportará un reporte llamado `reporte_pruebas.txt` (y su versión en Word `.docx`).

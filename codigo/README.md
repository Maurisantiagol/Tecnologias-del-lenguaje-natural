<div align="center">
  <h1>🍝 Asistente Virtual para Restaurante Italiano</h1>
  <p><strong>Un Chatbot End-to-End con NLP, Detección de Intenciones y Traducción en Tiempo Real</strong></p>

  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)
  ![spaCy](https://img.shields.io/badge/NLP-spaCy-blueviolet.svg)
</div>

---

Este proyecto es una solución integral orientada a la industria gastronómica. Consiste en un agente conversacional inteligente capaz de gestionar reservaciones, consultar menús, identificar alérgenos y hacer recomendaciones gastronómicas utilizando modelos de Machine Learning (NLP).

## 🌟 Características Principales

- **🤖 Reconocimiento de Intenciones (NLU):** Utiliza un clasificador **SVM (Support Vector Machine)** entrenado sobre representaciones TF-IDF para detectar de forma natural las peticiones del usuario (Ej: `Book_Table`, `Query_Menu`, `Recommend_Food`).
- **🎯 Sistema de Recomendación de Platillos:** Un motor basado en *Cosine Similarity* evalúa las consultas contra una base de datos pre-filtrada y procesada. Puede respetar restricciones dietéticas complejas (vegano, sin gluten, sin nueces, etc.).
- **🌍 Middleware de Traducción Transparente:** La arquitectura del modelo base está optimizada para inglés. Para solucionar la barrera del idioma, se integró `deep-translator`. El sistema recibe entradas en español, las procesa internamente en el idioma original del modelo, y entrega la salida generada de regreso en español en tiempo real.
- **🔌 API REST Estructurada:** Todo el núcleo de IA está encapsulado detrás de un servidor Flask, separando la lógica del frontend.

## 🏗️ Arquitectura del Sistema

El proyecto está diseñado bajo un esquema modular para facilitar su mantenimiento y escalabilidad:

- 🟢 **`app.py` (Punto de Entrada / Servidor)**: Este es el **archivo principal de ejecución**. Levanta el servidor Flask, expone los endpoints de comunicación (`/api/chat`, `/api/suggest`) y maneja el flujo de traducción de idiomas.
- 🧠 **`chatbot.py` / `chatbot_italiano.py` (Lógica de IA)**: Son los módulos internos que procesan el lenguaje natural. Son importados y utilizados por `app.py`. Se encargan de la lematización con spaCy y la predicción del SVM.
- 🎨 **`frontend/`**: Contiene la interfaz gráfica al usuario final (Vanilla HTML/CSS/JS) diseñada para ser estética y funcional, conectándose dinámicamente con `app.py`.
- 🧪 **`test_ruido_traduccion.py`**: Suite de pruebas robustas que simula escenarios reales donde el usuario comete errores ortográficos y de tipeo para evaluar la fiabilidad del modelo bajo estrés.

---

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos para desplegar el proyecto localmente de principio a fin.

### 1. Requisitos Previos e Instalación

Clona este repositorio o asegúrate de estar posicionado en la carpeta principal del código (`Codigo/`). Se recomienda crear un entorno virtual de Python.

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno virtual
venv\Scripts\activate   # En Windows
source venv/bin/activate # En Mac/Linux

# Instalar todas las dependencias
pip install -r requirements.txt
```

> **Nota:** El archivo `requirements.txt` ya se encarga de descargar el corpus necesario de `spaCy` (`en_core_web_sm`).

### 2. Ejecutar el Servidor (Backend)

> ⚠️ **IMPORTANTE:** El archivo que **siempre** debes ejecutar para poner en marcha el proyecto es `app.py`. No ejecutes `chatbot.py` directamente; ese archivo es solo un módulo interno al que llama el servidor principal.

Inicia la API ejecutando en tu terminal:

```bash
python app.py
```

Al hacerlo, el servidor Flask inicializará los modelos `.pkl` en memoria y quedará en escucha en `http://0.0.0.0:5000` listo para procesar peticiones HTTP.

### 3. Visualizar la Interfaz Web (Frontend)

Una vez que el servidor backend esté corriendo, abre un explorador de archivos, entra a la carpeta `frontend/` y abre el archivo **`index.html`** directamente en tu navegador (Chrome, Edge, etc.). 

La web buscará automáticamente la conexión al backend de Flask en tu equipo local (`localhost:5000`) y podrás comenzar a chatear.

---

## 🧪 Suite de Pruebas de Estrés y Ruido

El proyecto incluye un mecanismo formal para verificar qué tan resistente es el chatbot a los errores de ortografía de los usuarios.

Para correr estas pruebas, ejecuta en otra ventana de la terminal:

```bash
python test_ruido_traduccion.py
```

Este script generará automáticamente más de 200 iteraciones con diferentes niveles de error humano (ruido). Al finalizar, generará automáticamente **dos reportes detallados** (`reporte_pruebas.txt` y `reporte_pruebas.docx`) que documentan la precisión de las intenciones esperadas contra las obtenidas, incluyendo logs precisos de cómo actuó el traductor frente al texto deformado.

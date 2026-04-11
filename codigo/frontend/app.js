const chatDisplay = document.getElementById('chat-display');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const autocompleteContainer = document.getElementById('autocomplete-container');

const API_URL = 'https://tecnologias-del-lenguaje-natural.onrender.com';

// --- CHAT LOGIC ---

function appendMessage(sender, text, recipes = []) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    
    let contentHtml = `<div class="content">${text}</div>`;
    msgDiv.innerHTML = contentHtml;
    
    // Si hay recetas las añadiremos programáticamente para asignarles EventListeners
    if (recipes && recipes.length > 0) {
        const recipesContainer = document.createElement('div');
        recipesContainer.classList.add('recipes-container');
        
        recipes.forEach(r => {
            const card = document.createElement('div');
            card.classList.add('recipe-card');
            
            const matchPercent = Math.round(r.match * 100);
            const timeInfo = r.time > 0 ? `⏱ ${Math.round(r.time)} min` : '';
            const fastBadge = (r.time > 0 && r.time <= 30) ? `<span class="badge fast">🚀 Quick</span>` : '';
            
            // Construir la lista de instrucciones si existen
            let directionsHtml = '';
            if (r.directions && r.directions.length > 0) {
                const lis = r.directions.map(step => `<li>${step}</li>`).join('');
                directionsHtml = `
                    <div class="recipe-directions" style="display: none;">
                        <h4>👨‍🍳 How to cook it:</h4>
                        <ul>${lis}</ul>
                    </div>
                    <button class="expand-btn">Show Recipe</button>
                `;
            }

            card.innerHTML = `
                <div class="recipe-header">
                    <div class="recipe-title">${r.title}</div>
                    ${fastBadge}
                </div>
                <div class="recipe-meta">${r.course} | Match: ${matchPercent}% | ${timeInfo}</div>
                <div class="recipe-ingredients"><strong>Ingredients:</strong> <br/>${r.ingredients}</div>
                ${directionsHtml}
            `;
            
            // Lógica de expansión
            if (directionsHtml !== '') {
                const btn = card.querySelector('.expand-btn');
                const dirDiv = card.querySelector('.recipe-directions');
                btn.addEventListener('click', () => {
                    if (dirDiv.style.display === 'none') {
                        dirDiv.style.display = 'block';
                        btn.textContent = 'Hide Recipe';
                    } else {
                        dirDiv.style.display = 'none';
                        btn.textContent = 'Show Recipe';
                    }
                    chatDisplay.scrollTop = chatDisplay.scrollHeight;
                });
            }
            
            recipesContainer.appendChild(card);
        });
        
        msgDiv.appendChild(recipesContainer);
    }
    
    chatDisplay.appendChild(msgDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

async function sendMessage(text) {
    if (!text.trim()) return;
    
    // Add User msg
    appendMessage('user', text);
    userInput.value = '';
    autocompleteContainer.style.display = 'none';
    
    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.classList.add('message', 'bot-message');
    loadingDiv.innerHTML = `<div class="content"><em>Typing...</em></div>`;
    chatDisplay.appendChild(loadingDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        
        // Remove loading indicator
        document.getElementById(loadingId).remove();
        
        if (data.error) {
            appendMessage('bot', "Scusa, we are having technical difficulties.");
        } else {
            appendMessage('bot', data.reply, data.recipes);
        }
        
    } catch (error) {
        document.getElementById(loadingId).remove();
        console.error(error);
        appendMessage('bot', "Connection error. Make sure the server is running on port 5000.");
    }
}

sendBtn.addEventListener('click', () => sendMessage(userInput.value));
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage(userInput.value);
});


// --- AUTOCOMPLETE LOGIC ---

let autocompleteTimeout = null;

userInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    
    clearTimeout(autocompleteTimeout);
    
    if (query.length < 2) {
        autocompleteContainer.style.display = 'none';
        return;
    }
    
    autocompleteTimeout = setTimeout(async () => {
        try {
            const response = await fetch(`${API_URL}/suggest?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            if (data.suggestions && data.suggestions.length > 0) {
                renderSuggestions(data.suggestions);
            } else {
                autocompleteContainer.style.display = 'none';
            }
        } catch (e) {
            // Silently fail autocomplete on error
        }
    }, 300); // 300ms debounce
});

function renderSuggestions(suggestions) {
    autocompleteContainer.innerHTML = '';
    
    // De-duplicate
    const unique = [...new Set(suggestions)].slice(0, 5);
    
    unique.forEach(sug => {
        const div = document.createElement('div');
        div.classList.add('suggestion-item');
        div.textContent = sug;
        div.addEventListener('click', () => {
            userInput.value = sug;
            sendMessage(sug);
        });
        autocompleteContainer.appendChild(div);
    });
    
    autocompleteContainer.style.display = 'block';
}

// Close autocomplete when clicking outside
document.addEventListener('click', (e) => {
    if (!userInput.contains(e.target) && !autocompleteContainer.contains(e.target)) {
        autocompleteContainer.style.display = 'none';
    }
});

/* ============================================================
   app.js — Tilin's Kitchen Chat Frontend
   Conecta con Flask API (chatbot_italiano.py pipeline)
   ============================================================ */

'use strict';

const API_URL = 'https://tecnologias-del-lenguaje-natural.onrender.com/api';

// ── DOM References ─────────────────────────────────────────────
const chatDisplay          = document.getElementById('chat-display');
const userInput            = document.getElementById('user-input');
const sendBtn              = document.getElementById('send-btn');
const autocompleteContainer = document.getElementById('autocomplete-container');
const resetBtn             = document.getElementById('reset-btn');

// ── Intent Config (label + CSS class) ─────────────────────────
const INTENT_MAP = {
    'Book_Table'        : { label: '📅 Reservation',  cls: 'book'        },
    'Query_Menu'        : { label: '📋 Menu',          cls: 'menu'        },
    'Recommend_Food'    : { label: '🍝 Recommendation', cls: 'food'       },
    'Query_Ingredients' : { label: '🔍 Ingredients',   cls: 'ingredients' },
    'Modify_Booking'    : { label: '✏️ Modify Booking', cls: 'modify'     },
    'Discover_Food'     : { label: '🍝 Recommendation', cls: 'food'       },
    'Unknown'           : { label: '❓ Unknown',        cls: 'unknown'    },
};

// ── Utility: parse simple markdown (bold/italic) ───────────────
function parseMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g,     '<em>$1</em>')
        .replace(/\n/g,             '<br>');
}

// ── Append a message to the chat ───────────────────────────────
function appendMessage(sender, text, { recipes = [], intent = '' } = {}) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

    const bodyDiv = document.createElement('div');
    bodyDiv.classList.add('message-body');

    if (sender === 'bot') {
        // Avatar
        const avatar = document.createElement('div');
        avatar.classList.add('bot-avatar');
        avatar.setAttribute('aria-hidden', 'true');
        avatar.textContent = '👨‍🍳';
        msgDiv.appendChild(avatar);

        // Intent badge
        if (intent && INTENT_MAP[intent]) {
            const badge = document.createElement('span');
            badge.classList.add('intent-badge', INTENT_MAP[intent].cls);
            badge.textContent = INTENT_MAP[intent].label;
            bodyDiv.appendChild(badge);
        }
    }

    // Text content
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('content');
    contentDiv.innerHTML = parseMarkdown(text);
    bodyDiv.appendChild(contentDiv);

    // Recipe cards
    if (recipes && recipes.length > 0) {
        bodyDiv.appendChild(buildRecipesContainer(recipes));
    }

    msgDiv.appendChild(bodyDiv);
    chatDisplay.appendChild(msgDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

// ── Build recipes container ─────────────────────────────────────
function buildRecipesContainer(recipes) {
    const container = document.createElement('div');
    container.classList.add('recipes-container');

    recipes.forEach(r => {
        const card = document.createElement('div');
        card.classList.add('recipe-card');

        const matchPct   = Math.round((r.match || 0) * 100);
        const isQuick    = r.time > 0 && r.time <= 30;
        const timeStr    = r.time > 0 ? `⏱ ${Math.round(r.time)} min` : '';

        // ── Header Row ──────────────────────────────────────────
        const header = document.createElement('div');
        header.classList.add('recipe-header');

        const titleEl = document.createElement('div');
        titleEl.classList.add('recipe-title');
        titleEl.textContent = r.title;
        header.appendChild(titleEl);

        card.appendChild(header);

        // ── Dietary Badges Row ───────────────────────────────────
        const allBadges = [];
        if (isQuick) allBadges.push({ label: '🚀 Quick', cls: 'fast' });

        // From dietary_badges array (returned by backend)
        if (Array.isArray(r.dietary_badges)) {
            r.dietary_badges.forEach(b => {
                const cls = b.label.toLowerCase().replace(' ', '-').replace('-free', '-free');
                allBadges.push({ label: `${b.icon} ${b.label}`, cls });
            });
        }

        // Nut warning
        if (r.has_nuts) {
            allBadges.push({ label: '⚠️ Contains Nuts', cls: 'nuts' });
        }

        if (allBadges.length > 0) {
            const badgesRow = document.createElement('div');
            badgesRow.classList.add('badges-row');
            allBadges.forEach(b => {
                const span = document.createElement('span');
                span.classList.add('badge', b.cls);
                span.textContent = b.label;
                badgesRow.appendChild(span);
            });
            card.appendChild(badgesRow);
        }

        // ── Meta Row ─────────────────────────────────────────────
        const meta = document.createElement('div');
        meta.classList.add('recipe-meta');
        meta.innerHTML = `
            ${r.course ? `<span class="meta-item">🍽️ ${r.course}</span>` : ''}
            ${timeStr  ? `<span class="meta-item">${timeStr}</span>`      : ''}
            <span class="match-bar-container">
                <span style="font-size:0.75rem;color:var(--text-muted)">Match:</span>
                <span class="match-bar" title="${matchPct}% relevance">
                    <span class="match-bar-fill" style="width:0%"></span>
                </span>
                <span style="font-size:0.75rem;color:var(--accent-gold);font-weight:600">${matchPct}%</span>
            </span>
        `;
        card.appendChild(meta);

        // Animate match bar after render
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const fill = meta.querySelector('.match-bar-fill');
                if (fill) fill.style.width = `${matchPct}%`;
            });
        });

        // ── Ingredients ───────────────────────────────────────────
        if (r.ingredients) {
            const ingDiv = document.createElement('div');
            ingDiv.classList.add('recipe-ingredients');
            // Trim long ingredient lists
            const maxLen = 300;
            const ingText = r.ingredients.length > maxLen
                ? r.ingredients.slice(0, maxLen) + '…'
                : r.ingredients;
            ingDiv.innerHTML = `<strong>Ingredients:</strong><br>${ingText}`;
            card.appendChild(ingDiv);
        }

        // ── Directions (expandable) ───────────────────────────────
        if (r.directions && r.directions.length > 0) {
            const dirDiv = document.createElement('div');
            dirDiv.classList.add('recipe-directions');
            dirDiv.style.display = 'none';

            const h4 = document.createElement('h4');
            h4.textContent = '👨‍🍳 How to cook it:';
            dirDiv.appendChild(h4);

            const ol = document.createElement('ol');
            r.directions.forEach(step => {
                const li = document.createElement('li');
                li.textContent = step;
                ol.appendChild(li);
            });
            dirDiv.appendChild(ol);
            card.appendChild(dirDiv);

            const expandBtn = document.createElement('button');
            expandBtn.classList.add('expand-btn');
            expandBtn.innerHTML = '📖 Show Recipe Steps';
            expandBtn.addEventListener('click', () => {
                const isHidden = dirDiv.style.display === 'none';
                dirDiv.style.display = isHidden ? 'block' : 'none';
                expandBtn.innerHTML = isHidden ? '🙈 Hide Steps' : '📖 Show Recipe Steps';
                // Scroll so new content is visible
                setTimeout(() => { chatDisplay.scrollTop = chatDisplay.scrollHeight; }, 100);
            });
            card.appendChild(expandBtn);
        }

        container.appendChild(card);
    });

    return container;
}

// ── Typing indicator ────────────────────────────────────────────
let typingEl = null;

function showTyping() {
    removeTyping();
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'bot-message');

    const avatar = document.createElement('div');
    avatar.classList.add('bot-avatar');
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = '👨‍🍳';

    const bodyDiv = document.createElement('div');
    bodyDiv.classList.add('message-body');

    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.setAttribute('aria-label', 'Chef Bot is typing...');
    indicator.innerHTML = `
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;
    bodyDiv.appendChild(indicator);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bodyDiv);
    chatDisplay.appendChild(msgDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
    typingEl = msgDiv;
}

function removeTyping() {
    if (typingEl) {
        typingEl.remove();
        typingEl = null;
    }
}

// ── Send message ────────────────────────────────────────────────
async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    // Hide chips after first interaction
    const chips = document.getElementById('quick-chips');
    if (chips) chips.style.display = 'none';

    // User bubble
    appendMessage('user', trimmed);
    userInput.value = '';
    hideAutocomplete();

    // Disable input while waiting
    setInputEnabled(false);
    showTyping();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: trimmed }),
        });

        removeTyping();

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            appendMessage('bot', `⚠️ Server error: ${err.error || response.statusText}`, {});
            return;
        }

        const data = await response.json();

        if (data.error) {
            appendMessage('bot', `⚠️ ${data.error}`, {});
        } else {
            appendMessage('bot', data.reply || '', {
                recipes: data.recipes || [],
                intent: data.intent  || '',
            });
        }

    } catch (err) {
        removeTyping();
        console.error('[Chef Bot] Fetch error:', err);
        appendMessage('bot',
            '🔌 Connection error. Make sure the Flask server is running on port 5000.\n\n' +
            '<em>Start it with:</em> <code>python app.py</code>', {});
    } finally {
        setInputEnabled(true);
        userInput.focus();
    }
}

function setInputEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled   = !enabled;
    sendBtn.style.opacity = enabled ? '1' : '0.5';
}

// ── Reset conversation ──────────────────────────────────────────
async function resetConversation() {
    try {
        await fetch(`${API_URL}/reset`, { method: 'POST' });
    } catch (_) { /* server may be down, still reset UI */ }

    // Clear chat display except welcome message rebuild
    chatDisplay.innerHTML = '';

    // Re-create welcome message
    buildWelcomeMessage();
}

function buildWelcomeMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'bot-message');
    msgDiv.id = 'welcome-msg';

    const avatar = document.createElement('div');
    avatar.classList.add('bot-avatar');
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = '👨‍🍳';

    const bodyDiv = document.createElement('div');
    bodyDiv.classList.add('message-body');

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('content');
    contentDiv.innerHTML = `
        <strong>Buongiorno!</strong> I'm Chef Bot, your personal Italian dining guide. 🍕<br><br>
        I can help you with:
        <ul class="welcome-list">
            <li>🍝 <strong>Recipe recommendations</strong> — including dietary needs</li>
            <li>📋 <strong>Browse our menu</strong> categories</li>
            <li>📅 <strong>Book or manage</strong> your table reservation</li>
            <li>🌱 <strong>Dietary filters</strong> — vegan, gluten-free, nut-free & more</li>
        </ul>
    `;
    bodyDiv.appendChild(contentDiv);

    // Rebuild chips
    const chipsDiv = document.createElement('div');
    chipsDiv.classList.add('quick-chips');
    chipsDiv.id = 'quick-chips';

    const chipDefs = [
        { msg: "What's on the menu?",           label: '📋 Ver menú' },
        { msg: 'Recommend me a pasta dish',      label: '🍝 Pasta'    },
        { msg: 'I want a quick vegan option',    label: '🌱 Vegan'    },
        { msg: 'Book a table for Friday at 8pm', label: '📅 Reservar' },
        { msg: 'Something gluten free please',   label: '🌾 Gluten-Free' },
    ];

    chipDefs.forEach(({ msg, label }) => {
        const btn = document.createElement('button');
        btn.classList.add('chip');
        btn.dataset.msg = msg;
        btn.textContent = label;
        btn.addEventListener('click', () => sendMessage(msg));
        chipsDiv.appendChild(btn);
    });

    bodyDiv.appendChild(chipsDiv);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bodyDiv);
    chatDisplay.appendChild(msgDiv);
}

// ── Event Listeners ─────────────────────────────────────────────

sendBtn.addEventListener('click', () => sendMessage(userInput.value));
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(userInput.value);
    }
});

resetBtn.addEventListener('click', resetConversation);

// ── Initial quick chips (from HTML) ────────────────────────────
document.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => sendMessage(btn.dataset.msg));
});

// ── Autocomplete ────────────────────────────────────────────────
let autocompleteTimeout = null;

userInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    clearTimeout(autocompleteTimeout);

    if (query.length < 2) {
        hideAutocomplete();
        return;
    }

    autocompleteTimeout = setTimeout(async () => {
        try {
            const res  = await fetch(`${API_URL}/suggest?q=${encodeURIComponent(query)}`);
            const data = await res.json();
            if (data.suggestions && data.suggestions.length > 0) {
                renderSuggestions(data.suggestions);
            } else {
                hideAutocomplete();
            }
        } catch (_) {
            hideAutocomplete();
        }
    }, 280);
});

function renderSuggestions(suggestions) {
    autocompleteContainer.innerHTML = '';
    const unique = [...new Set(suggestions)].slice(0, 6);

    unique.forEach(sug => {
        const div = document.createElement('div');
        div.classList.add('suggestion-item');
        div.setAttribute('role', 'option');
        div.textContent = sug;
        div.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent blur before click
            userInput.value = sug;
            sendMessage(sug);
        });
        autocompleteContainer.appendChild(div);
    });

    autocompleteContainer.style.display = 'block';
}

function hideAutocomplete() {
    autocompleteContainer.style.display = 'none';
}

document.addEventListener('click', (e) => {
    if (!userInput.contains(e.target) && !autocompleteContainer.contains(e.target)) {
        hideAutocomplete();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideAutocomplete();
});

// ── Background Particles (decorative) ──────────────────────────
function initParticles() {
    const container = document.getElementById('bg-particles');
    if (!container) return;

    const emojis = ['🍕', '🍝', '🧄', '🫒', '🍷', '🧀', '🌿', '🍅'];
    const count  = 12;

    for (let i = 0; i < count; i++) {
        const p = document.createElement('div');
        p.classList.add('particle');
        const size = Math.random() * 30 + 20;
        p.style.cssText = `
            width:  ${size}px;
            height: ${size}px;
            left:   ${Math.random() * 100}%;
            font-size: ${size * 0.7}px;
            animation-duration: ${Math.random() * 25 + 20}s;
            animation-delay:    ${Math.random() * -30}s;
            background: transparent;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.04;
            filter: blur(1px);
        `;
        p.textContent = emojis[Math.floor(Math.random() * emojis.length)];
        container.appendChild(p);
    }
}

initParticles();

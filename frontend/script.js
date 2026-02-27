// -- CONFIGURAÇÃO --
// Detecta se estamos rodando no Live Server (5500) e aponta para o backend (5000)
// Se já estivermos na 5000, usa caminhos relativos.
const API_BASE_URL = window.location.port === '5500' ? 'http://127.0.0.1:5000' : '';

// -- ESTADO GLOBAL --
let currentSessionId = localStorage.getItem('lastSessionId') || 'session_' + Date.now();
let isProcessing = false;

// -- INICIALIZAÇÃO --
document.addEventListener('DOMContentLoaded', () => {
    loadFiles();
    loadSessions().then(() => {
        loadHistory(currentSessionId);
    });

    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
        const toggle = document.getElementById('themeToggle');
        if (toggle) toggle.textContent = '☀️';
    }

    // Prevenir reload acidental se alguém colocar um <form> depois
    const questionInput = document.getElementById('questionInput');
    if (questionInput) {
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});

function showWelcomeMessage() {
    const container = document.getElementById('chatMessages');
    if (container && container.children.length === 0) {
        addMessage("Olá! Sou seu assistente inteligente. Carregue um PDF, TXT ou uma imagem para começarmos a conversar sobre eles!", 'bot');
    }
}

// -- SESSÕES --
async function loadSessions() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/sessions`);
        const data = await res.json();
        renderSessions(data.sessions || []);
    } catch (err) {
        console.error("Erro ao carregar sessões:", err);
    }
}

function renderSessions(sessions) {
    const list = document.getElementById('sessionsList');
    if (!list) return;
    list.innerHTML = '';

    sessions.forEach(session => {
        const div = document.createElement('div');
        div.className = `session-item ${session.id === currentSessionId ? 'active' : ''}`;
        div.innerHTML = `<span>${session.title || 'Nova Conversa'}</span>`;
        div.onclick = () => switchSession(session.id);
        list.appendChild(div);
    });
}

async function loadHistory(sessionId) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    container.innerHTML = '';

    try {
        const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`);
        const data = await res.json();

        if (data.history && data.history.length > 0) {
            data.history.forEach(msg => {
                addMessage(msg.text, msg.sender);
            });
        } else {
            showWelcomeMessage();
        }
    } catch (err) {
        console.error("Erro ao carregar histórico:", err);
        showWelcomeMessage();
    }
}

function startNewChat() {
    if (isProcessing) return;
    currentSessionId = 'session_' + Date.now();
    localStorage.setItem('lastSessionId', currentSessionId);
    document.getElementById('chatMessages').innerHTML = '';
    showWelcomeMessage();
    loadSessions();
}

async function switchSession(id) {
    if (isProcessing || id === currentSessionId) return;
    currentSessionId = id;
    localStorage.setItem('lastSessionId', id);
    await loadHistory(id);
    loadSessions();
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

// -- CHAT --
async function sendMessage() {
    const input = document.getElementById('questionInput');
    const text = input.value.trim();
    if (!text || isProcessing) return;

    addMessage(text, 'user');
    input.value = '';
    isProcessing = true;
    document.getElementById('sendBtn').disabled = true;

    const botMessageDiv = createMessageBubble('<span class="typing-dots">Analisando...</span>', 'bot');
    botMessageDiv.classList.add('streaming'); // Ativa o cursor piscante
    document.getElementById('chatMessages').appendChild(botMessageDiv);
    scrollToBottom();

    const contentDiv = botMessageDiv.querySelector('.message-content');
    let fullText = "";
    let typewriterQueue = [];
    let isTyping = false;

    // Função que consome a fila de caracteres para o efeito de digitação
    function processTypewriter() {
        if (typewriterQueue.length === 0) {
            isTyping = false;
            return;
        }
        isTyping = true;
        const char = typewriterQueue.shift();
        fullText += char;
        contentDiv.innerHTML = marked.parse(fullText);

        // Se a fila estiver muito cheia, acelera a digitação
        const delay = typewriterQueue.length > 50 ? 2 : 15;
        setTimeout(processTypewriter, delay);
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: text,
                session_id: currentSessionId
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || "Erro no servidor");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let isFirstToken = true;
        let streamBuffer = ""; // Buffer para acumular fragmentos da stream

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            streamBuffer += decoder.decode(value, { stream: true });

            // SSE separa mensagens por dois line-breaks (\n\n)
            let boundary = streamBuffer.indexOf('\n\n');
            while (boundary !== -1) {
                const rawMessage = streamBuffer.substring(0, boundary).trim();
                streamBuffer = streamBuffer.substring(boundary + 2);

                if (rawMessage.startsWith('data: ')) {
                    try {
                        const jsonStr = rawMessage.substring(6);
                        const data = JSON.parse(jsonStr);

                        if (data.token) {
                            if (isFirstToken) {
                                contentDiv.innerHTML = "";
                                isFirstToken = false;
                            }
                            // Efeito de digitação
                            typewriterQueue.push(...data.token.split(''));
                            if (!isTyping) processTypewriter();
                        }

                        if (data.done) {
                            // Finalizou a stream
                            break;
                        }
                    } catch (e) {
                        console.warn("Falha ao processar fragmento JSON:", e);
                    }
                }
                boundary = streamBuffer.indexOf('\n\n');
            }
            scrollToBottom();
        }

        botMessageDiv.classList.remove('streaming');
        loadSessions();
    } catch (err) {
        if (botMessageDiv) {
            botMessageDiv.classList.remove('streaming');
            const contentDiv = botMessageDiv.querySelector('.message-content');
            contentDiv.innerHTML = `<span style="color: #ef4444;">Erro: ${err.message}</span>`;
        }
        console.error(err);
    } finally {
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
        scrollToBottom();
    }
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    if (container) container.scrollTop = container.scrollHeight;
}

// -- UI HELPERS --
function addMessage(text, sender) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const bubble = createMessageBubble(text, sender);
    container.appendChild(bubble);
    scrollToBottom();
}

function createMessageBubble(text, sender) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;

    const avatar = sender === 'user' ? '👤' : '🤖';
    const name = sender === 'user' ? 'Você' : 'Assistente';

    div.innerHTML = `
        <div class="avatar ${sender}">${avatar}</div>
        <div class="message-body">
            <div class="message-header">${name}</div>
            <div class="message-content">${sender === 'user' ? text : marked.parse(text)}</div>
        </div>
    `;
    return div;
}

// -- UPLOADS & FILES --
async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    toggleProgress(true, "Processando documento...");
    try {
        const res = await fetch(`${API_BASE_URL}/api/upload`, { method: 'POST', body: formData });
        if (res.ok) {
            addMessage(`Documento "**${fileInput.files[0].name}**" processado com sucesso!`, 'bot');
            loadFiles();
        } else {
            const data = await res.json();
            alert("Erro: " + (data.error || "Erro desconhecido"));
        }
    } catch (err) {
        alert("Erro de conexão ao enviar documento.");
    } finally {
        toggleProgress(false);
        fileInput.value = '';
    }
}

async function uploadImage() {
    const fileInput = document.getElementById('imageFile');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    toggleProgress(true, "Analisando Imagem...");
    try {
        const res = await fetch(`${API_BASE_URL}/api/upload-image`, { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            addMessage(`🖼️ **Análise da Imagem:**\n\n${data.analysis}`, 'bot');
            loadFiles();
        } else {
            alert("Erro: " + (data.error || "Erro desconhecido"));
        }
    } catch (err) {
        alert("Erro de conexão ao enviar imagem.");
    } finally {
        toggleProgress(false);
        fileInput.value = '';
    }
}

async function loadFiles() {
    try {
        const [pdfRes, imgRes] = await Promise.all([
            fetch(`${API_BASE_URL}/api/pdfs`),
            fetch(`${API_BASE_URL}/api/images`)
        ]);
        const pdfs = await pdfRes.json();
        const images = await imgRes.json();

        const container = document.getElementById('fileList');
        if (!container) return;
        container.innerHTML = '';

        const allFiles = [
            ...(pdfs.pdfs || []).map(n => ({ name: n, type: 'pdf' })),
            ...(images.images || []).map(n => ({ name: n, type: 'image' }))
        ];

        if (allFiles.length === 0) {
            container.innerHTML = '<p style="font-size: 11px; color: var(--text-dim); text-align: center;">Nenhum arquivo.</p>';
            return;
        }

        allFiles.forEach(file => {
            const div = document.createElement('div');
            div.className = 'file-chip';
            div.innerHTML = `
                <span title="${file.name}">${file.type === 'pdf' ? '📄' : '🖼️'} ${file.name}</span>
                <span style="cursor: pointer; color: #ef4444; font-weight: bold;" onclick="deleteFile('${file.name}', '${file.type}')">×</span>
            `;
            container.appendChild(div);
        });
    } catch (err) { console.error("Erro ao listar arquivos", err); }
}

async function deleteFile(name, type) {
    if (!confirm(`Excluir ${name}?`)) return;
    const endpoint = type === 'pdf' ? `/api/pdfs/${name}` : `/api/images/${name}`;
    await fetch(`${API_BASE_URL}${endpoint}`, { method: 'DELETE' });
    loadFiles();
}

async function clearEverything() {
    if (!confirm("Limpar todo o banco de dados e memória?")) return;
    toggleProgress(true, "Limpando...");
    await fetch(`${API_BASE_URL}/api/clear`, { method: 'POST' });
    toggleProgress(false);
    loadFiles();
    startNewChat();
}

// -- MISC --
function toggleProgress(show, title = "Processando...") {
    const overlay = document.getElementById('progressOverlay');
    if (!overlay) return;
    const fill = document.getElementById('pFill');
    document.getElementById('progressTitle').textContent = title;
    overlay.style.display = show ? 'flex' : 'none';
    if (show && fill) {
        fill.style.width = '0%';
        setTimeout(() => fill.style.width = '90%', 100);
    }
}

function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    const isDark = body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    const toggle = document.getElementById('themeToggle');
    if (toggle) toggle.textContent = isDark ? '☀️' : '🌓';
}

function handleKeyPress(e) {
    // Esta função agora é secundária pois adicionei um listener no DOMContentLoaded
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
}

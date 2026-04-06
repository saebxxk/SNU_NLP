const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typingBox = document.getElementById('typing-box');
const clearBtn = document.getElementById('clear-btn');

const API_URL = 'http://localhost:8000';

// 텍스트 영역 높이 자동 조절
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
});

// 엔터 키 처리 (Shift+Enter는 줄바꿈)
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', clearHistory);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. 사용자 메시지 추가
    addMessage(text, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    // 2. 생각 중 표시
    showTyping(true);

    try {
        // 3. API 호출
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });

        if (!response.ok) throw new Error('서버 응답 오류');

        const data = await response.json();
        
        // 4. 생각 중 숨기기 및 봇 메시지 추가
        showTyping(false);
        addMessage(data.answer, 'bot');

    } catch (error) {
        console.error('Error:', error);
        showTyping(false);
        addMessage(`죄송합니다. 서버와 통신 중 오류가 발생했습니다. (백엔드 서버가 실행 중인지 확인해 주세요)`, 'bot');
    }
}

function addMessage(text, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = `<img src="chatbot_logo.png" alt="${role}">`;
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    // 개행 문자 처리 및 강조 표시
    let processedText = text.replace(/\n/g, '<br>');
    processedText = processedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    processedText = processedText.replace(/•\s?(.*?)(<br>|$)/g, '<li>$1</li>');
    if (processedText.includes('<li>')) {
        processedText = processedText.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    }

    bubble.innerHTML = processedText;

    if (role === 'bot') {
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
    } else {
        messageDiv.appendChild(bubble);
        // 사용자 아바타는 생략하거나 다른 것으로 대체 가능
    }

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showTyping(show) {
    if (show) {
        typingBox.style.display = 'flex';
        chatContainer.appendChild(typingBox);
        scrollToBottom();
    } else {
        typingBox.style.display = 'none';
        chatContainer.removeChild(typingBox);
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function clearHistory() {
    if (!confirm('정말로 대화 기록을 초기화하시겠습니까?')) return;

    try {
        await fetch(`${API_URL}/clear`, { method: 'POST' });
        
        // UI 초기화 (첫 환영 인사만 남김)
        const welcomeMsg = document.querySelector('.message.welcome');
        chatContainer.innerHTML = '';
        if (welcomeMsg) chatContainer.appendChild(welcomeMsg);
        
        alert('대화 기록이 초기화되었습니다.');
    } catch (error) {
        console.error('Error clearing history:', error);
    }
}

function useSuggestion(text) {
    userInput.value = text;
    userInput.focus();
    userInput.dispatchEvent(new Event('input'));
}

// 윈도우 전역 함수로 노출 (onclick용)
window.useSuggestion = useSuggestion;

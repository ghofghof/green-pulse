document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const typingIndicator = document.getElementById('typing-indicator');

    // Ajouter un message au chat
    function addMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        if (typeof content === 'string') {
            messageDiv.innerHTML = content.replace(/\n/g, '<br>');
        } else {
            messageDiv.innerHTML = formatBotResponse(content);
        }
        
        chatMessages.insertBefore(messageDiv, typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Formater la réponse du bot avec les anomalies
    function formatBotResponse(responseData) {
        let html = responseData.llm_response.replace(/\n/g, '<br>');
        
        if (responseData.matched_problem && responseData.anomalies) {
            const problem = responseData.matched_problem;
            
            html += `<div class="problem-info">
                <strong>Problème détecté :</strong> ${problem.Categorie} > ${problem['Sous-categorie']}
                <br><strong>Seuils appliqués :</strong> ${JSON.stringify(problem.Seuils)}
            </div>`;
            
            if (responseData.anomalies.length > 0) {
                html += '<p><strong>Équipements en anomalie :</strong></p>';
                html += '<table class="anomaly-table"><tr>';
                
                // En-têtes
                const cols = ['Equipment_ID', 'Equipment_Name', 'Supplier_Name', ...problem.Indicateurs];
                cols.forEach(col => {
                    html += `<th>${col}</th>`;
                });
                html += '</tr>';
                
                // Données
                responseData.anomalies.forEach(item => {
                    html += '<tr>';
                    cols.forEach(col => {
                        html += `<td>${item[col]}</td>`;
                    });
                    html += '</tr>';
                });
                
                html += '</table>';
            } else {
                html += '<p>Aucun équipement ne correspond aux critères d\'anomalie.</p>';
            }
        }
        
        return html;
    }

    // Envoyer le message au serveur
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        addMessage(message, true);
        userInput.value = '';
        
        // Afficher l'indicateur de typing
        typingIndicator.style.display = 'flex';
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message })
            });
            
            if (!response.ok) {
                throw new Error('Erreur du serveur');
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            addMessage(data, false);
        } catch (error) {
            addMessage(`Erreur: ${error.message}`, false);
        } finally {
            typingIndicator.style.display = 'none';
        }
    }

    // Événements
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Message de bienvenue
    setTimeout(() => {
        addMessage("Bonjour ! Je suis votre assistant expert en durabilité industrielle. Décrivez-moi votre problème et je vous fournirai des recommandations ainsi qu'une analyse des anomalies potentielles.", false);
    }, 500);
});
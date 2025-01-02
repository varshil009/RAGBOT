function uploadPDF() {
    // Your existing upload logic
    // After successful upload:
    $('#upload-section').hide();
    $('#chat-interface').show();
}

function askQuestion() {
    var question = $('#question-input').val();
    if (question.trim() === '') return;

    // Add user message to chat
    addMessage(question, 'user-message');

    // Clear input field
    $('#question-input').val('');

    // Send question to server and handle response
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({question: question})
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.answer, 'bot-message');
        if (data.image) {
            addImage(data.image);
        }
    })
    .catch(error => console.error('Error:', error));
}

function addMessage(text, className) {
    $('#chat-messages').append(`<div class="message ${className}">${text}</div>`);
    scrollToBottom();
}

function addImage(imageBase64) {
    $('#chat-messages').append(`<img src="data:image/png;base64,${imageBase64}" alt="Response Image" class="response-image">`);
    scrollToBottom();
}

function scrollToBottom() {
    var chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event listener for Enter key in input field
$('#question-input').keypress(function(e) {
    if(e.which == 13) {
        askQuestion();
    }
});

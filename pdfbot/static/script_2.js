function uploadPDF() {
    const fileInput = document.getElementById('pdf-upload');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    document.getElementById('processing-message').style.display = 'block';
    document.getElementById('upload-section').style.display = 'none';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'PDF uploaded and processed successfully') {
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('chat-interface').style.display = 'block';
            document.getElementById('pdf-viewer').src = URL.createObjectURL(fileInput.files[0]);
        }
    })
    .catch(error => console.error('Error:', error));
}

function askQuestion() {
    const question = document.getElementById('question-input').value;
    if (!question) return;

    addMessage(question, 'user-message');
    document.getElementById('question-input').value = '';

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
        if (data.page_numbers && data.page_numbers.length > 0) {
            document.getElementById('pdf-viewer').contentWindow.PDFViewerApplication.page = data.page_numbers[0] + 1;
        }
    })
    .catch(error => console.error('Error:', error));
}

function addMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.innerHTML = text;
    document.getElementById('chat-messages').appendChild(messageDiv);
    messageDiv.scrollIntoView({behavior: "smooth"});
}

function addImage(imageBase64) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.style.maxWidth = '100%';
    document.getElementById('chat-messages').appendChild(img);
    img.scrollIntoView({behavior: "smooth"});
}

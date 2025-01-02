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
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Backend Response:', data); // Debugging
        if (data.answer) {
            addMessage(data.answer, 'bot-message', true); // Render Markdown
        } else {
            console.error('No answer field in response:', data);
        }
        if (data.image) {
            addImage(data.image);
        }
        if (data.page_numbers && data.page_numbers.length > 0) {
            addPageNavigation(data.page_numbers);
        }
    })
    .catch(error => console.error('Error:', error));
}

function addMessage(text, className, isMarkdown = false) {
    if (!text) {
        console.error('Empty or undefined text passed to addMessage:', text);
        return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;

    try {
        if (isMarkdown) {
            if (typeof marked !== 'undefined') {
                messageDiv.innerHTML = marked.parse(text); // Render Markdown
            } else {
                console.warn('Markdown library is not loaded. Displaying as plain text.');
                messageDiv.textContent = text;
            }
        } else {
            messageDiv.textContent = text; // Plain text fallback
        }
    } catch (error) {
        console.error('Error rendering message:', error);
        messageDiv.textContent = text; // Fallback in case of error
    }

    document.getElementById('chat-messages').appendChild(messageDiv);
    messageDiv.scrollIntoView({ behavior: "smooth" });
}

function addImage(imageBase64) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.style.maxWidth = '100%';
    document.getElementById('chat-messages').appendChild(img);
    img.scrollIntoView({ behavior: "smooth" });
}

function addPageNavigation(pageNumbers) {
    const navContainer = document.createElement('div');
    navContainer.className = 'page-navigation';

    pageNumbers.forEach(page => {
        const button = document.createElement('button');
        button.textContent = `Page ${page}`;
        button.onclick = () => jumpToPage(page);
        navContainer.appendChild(button);
    });

    document.getElementById('chat-messages').appendChild(navContainer);
    navContainer.scrollIntoView({ behavior: "smooth" });
}

function jumpToPage(pageNumber) {
    const viewer = document.getElementById('pdf-viewer');
    viewer.contentWindow.PDFViewerApplication.page = pageNumber;
}

// Add these at the start of script.js
let pdfViewerReady = false;
let pdfViewer = null;

window.addEventListener('message', function(event) {
    // Store the PDF viewer reference when it's ready
    if (event.data && event.data.type === 'READY') {
        pdfViewer = document.getElementById('pdf-viewer').contentWindow;
    }
});

// Modified uploadPDF function
function uploadPDF() {
    const fileInput = document.getElementById('pdf-upload');
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file first');
        return;
    }

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
            
            const pdfUrl = URL.createObjectURL(fileInput.files[0]);
            const viewer = document.getElementById('pdf-viewer');
            
            // Set up PDF viewer with event listener
            viewer.src = `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(pdfUrl)}`;
            
            // Add load event listener to the iframe
            viewer.onload = function() {
                // Wait for PDF.js to fully initialize
                const checkPDFApplication = setInterval(() => {
                    if (viewer.contentWindow.PDFViewerApplication) {
                        clearInterval(checkPDFApplication);
                        pdfViewer = viewer.contentWindow;
                        pdfViewerReady = true;
                    }
                }, 100);
            };
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('processing-message').style.display = 'none';
        document.getElementById('upload-section').style.display = 'block';
        alert('Error uploading PDF. Please try again.');
    });
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
        if (data.answer) {
            addMessage(data.answer, 'bot-message', true);
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
    if (!text) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;

    if (isMarkdown && typeof marked !== 'undefined') {
        messageDiv.innerHTML = marked.parse(text);
    } else {
        messageDiv.textContent = text;
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

// Modified addPageNavigation function
function addPageNavigation(pageNumbers) {
    const navContainer = document.createElement('div');
    navContainer.className = 'page-navigation';

    pageNumbers.forEach(page => {
        const button = document.createElement('button');
        button.textContent = `Page ${page}`;
        button.onclick = () => {
            if (pdfViewerReady) {
                jumpToPage(page);
            } else {
                console.log('PDF viewer not ready yet');
            }
        };
        navContainer.appendChild(button);
    });

    document.getElementById('chat-messages').appendChild(navContainer);
    navContainer.scrollIntoView({ behavior: "smooth" });
}

// Modified jumpToPage function
function jumpToPage(pageNumber) {
    const viewer = document.getElementById('pdf-viewer');
    if (!viewer) return;

    // Convert pageNumber to integer
    pageNumber = parseInt(pageNumber);
    
    try {
        // Try multiple methods to change page
        viewer.contentWindow.postMessage({
            type: 'jumpToPage',
            pageNumber: pageNumber   // PDF.js uses 0-based page numbers
        }, '*');

        // Also try direct PDFViewerApplication access
        if (viewer.contentWindow.PDFViewerApplication) {
            viewer.contentWindow.PDFViewerApplication.page = pageNumber;
        }
    } catch (error) {
        console.error('Error jumping to page:', error);
    }
}

// Add event listener for Enter key in question input
document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});

// Add this event listener
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'pdfViewer-ready') {
        pdfViewerReady = true;
        pdfViewer = document.getElementById('pdf-viewer').contentWindow;
    }
});
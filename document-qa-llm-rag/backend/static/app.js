const uploadForm = document.getElementById("upload-form");
const questionForm = document.getElementById("question-form");
const fileInput = document.getElementById("file-input");
const questionInput = document.getElementById("question-input");
const toast = document.getElementById("toast");
const resetWorkspaceButton = document.getElementById("reset-workspace");
const clearHistoryButton = document.getElementById("clear-history");

const statFile = document.getElementById("stat-file");
const statCharacters = document.getElementById("stat-characters");
const statChunks = document.getElementById("stat-chunks");
const documentPreview = document.getElementById("document-preview");
const answerText = document.getElementById("answer-text");
const answerDocument = document.getElementById("answer-document");
const answerConfidence = document.getElementById("answer-confidence");
const historyList = document.getElementById("history-list");
const chips = document.querySelectorAll(".chip");

function showToast(message, isError = false) {
    toast.hidden = false;
    toast.textContent = message;
    toast.style.background = isError ? "rgba(126, 33, 33, 0.95)" : "rgba(27, 26, 23, 0.92)";

    window.clearTimeout(showToast.timeoutId);
    showToast.timeoutId = window.setTimeout(() => {
        toast.hidden = true;
    }, 3200);
}

function setDefaultAnswerState() {
    answerDocument.textContent = "";
    answerConfidence.textContent = "Confidence: waiting";
    answerText.textContent = "Answers will appear here once a document is loaded.";
}

function setDefaultDocumentState() {
    statFile.textContent = "No document yet";
    statCharacters.textContent = "0";
    statChunks.textContent = "0";
    documentPreview.textContent = "Your document preview will appear here after upload.";
    setDefaultAnswerState();
}

function renderHistory(items) {
    historyList.innerHTML = "";

    if (!items.length) {
        const empty = document.createElement("p");
        empty.className = "history-empty";
        empty.textContent = "Your recent questions and answers will appear here.";
        historyList.appendChild(empty);
        return;
    }

    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "history-card";
        card.innerHTML = `
            <div class="history-top">
                <strong>${item.question}</strong>
                <span class="history-mode">${item.mode}</span>
            </div>
            <p>${item.answer}</p>
            <div class="history-bottom">
                <span>Confidence: ${item.confidence}</span>
            </div>
        `;
        historyList.appendChild(card);
    });
}

async function fetchHistory() {
    try {
        const response = await fetch("/api/history");
        const data = await response.json();
        renderHistory(data.items || []);
    } catch (error) {
        showToast("Could not load question history.", true);
    }
}

async function fetchHealth() {
    try {
        await fetch("/api/health");
    } catch (error) {
        showToast("Backend is not responding yet.", true);
    }
}

async function resetWorkspace(showMessage = true) {
    try {
        const response = await fetch("/api/reset", {
            method: "POST",
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Reset failed.");
        }

        uploadForm.reset();
        questionForm.reset();
        setDefaultDocumentState();
        renderHistory([]);

        if (showMessage) {
            showToast(data.message || "Workspace cleared.");
        }
    } catch (error) {
        showToast(error.message, true);
    }
}

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        showToast("Choose a PDF first.", true);
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const button = uploadForm.querySelector("button");
    button.disabled = true;
    button.textContent = "Processing...";

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Upload failed.");
        }

        statFile.textContent = data.filename;
        statCharacters.textContent = data.total_characters.toLocaleString();
        statChunks.textContent = data.total_chunks.toLocaleString();
        documentPreview.textContent = data.preview || "No preview available.";
        answerDocument.textContent = data.filename;
        answerConfidence.textContent = "Confidence: waiting";
        answerText.textContent = "Document ready. Ask a concise question to get a grounded answer.";
        renderHistory([]);
        await fetchHealth();
        showToast("Document processed successfully.");
    } catch (error) {
        showToast(error.message, true);
    } finally {
        button.disabled = false;
        button.textContent = "Process document";
    }
});

questionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question) {
        showToast("Enter a question first.", true);
        return;
    }

    const button = questionForm.querySelector("button");
    button.disabled = true;
    button.textContent = "Thinking...";

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Question failed.");
        }

        answerText.textContent = data.answer;
        const mode = data.generation?.mode ? ` | ${data.generation.mode}` : "";
        answerDocument.textContent = `${data.document || ""}${mode}`;
        answerConfidence.textContent = `Confidence: ${data.confidence || "unknown"}`;
        await fetchHistory();
        showToast("Answer generated.");
    } catch (error) {
        showToast(error.message, true);
    } finally {
        button.disabled = false;
        button.textContent = "Generate answer";
    }
});

resetWorkspaceButton.addEventListener("click", () => {
    resetWorkspace(true);
});

clearHistoryButton.addEventListener("click", async () => {
    await resetWorkspace(true);
});

chips.forEach((chip) => {
    chip.addEventListener("click", () => {
        questionInput.value = chip.dataset.question || "";
        questionInput.focus();
    });
});

setDefaultDocumentState();
fetchHealth();
fetchHistory();

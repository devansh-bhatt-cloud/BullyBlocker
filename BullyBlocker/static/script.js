document.addEventListener('DOMContentLoaded', function () {
    const sendBtn = document.getElementById("send-btn");
    const messageInput = document.getElementById("message-input");
    const resultBox = document.getElementById("result-box");

    sendBtn.addEventListener("click", async () => {
        const message = messageInput.value.trim();
        if (!message) return;

        const formData = new FormData();
        formData.append("message", message);

        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        resultBox.innerText = data.result;
        resultBox.style.display = "block";
    });
});

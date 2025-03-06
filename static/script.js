// Wait for the document to fully load
document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-btn");
    const chatBox = document.getElementById("chat-box");
    
    // Add event listener for send button
    sendButton.addEventListener("click", sendMessage);
    
    // Add event listener for Enter key
    userInput.addEventListener("keyup", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
   
    // Function to send message
    async function sendMessage() {
        const messageText = userInput.value;
        
        if (messageText.trim() === "") return;
        
        // Add user message to chat
        addMessage(messageText, 'user-message');
        
        // Clear input field
        userInput.value = "";
        
        // Show typing indicator
        const typingIndicator = addMessage("...", 'bot-message typing');
        
        try {
            // Send request to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });
            // console.log("clear ha sarfarz")
            // Get response from server
            const data = await response.json();
            
            // Remove typing indicator
            chatBox.removeChild(typingIndicator);
            
            // Add bot response to chat
            addMessage(data.response, 'bot-message');
        }  
    
        catch (error) {
            // Remove typing indicator
            chatBox.removeChild(typingIndicator);
            
            // Add error message
            addMessage("معاف ڪجو، رابطي ۾ خرابي آئي آهي!", 'bot-message error');
            console.error('Error:', error);
        }
    }
    
    // Function to add message to chat
    function addMessage(text, className) {
        const messageElement = document.createElement("div");
        messageElement.className = className;
        messageElement.textContent = text;
        chatBox.appendChild(messageElement);
        
        // Scroll to bottom
        chatBox.scrollTop = chatBox.scrollHeight;
        
        return messageElement;
    }
});
console.log("model succesfull")
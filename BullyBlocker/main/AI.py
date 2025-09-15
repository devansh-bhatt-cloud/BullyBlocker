import nltk
import os
import json
import random
from detoxify import Detoxify
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tkinter as tk # Import tkinter for GUI
import threading # Import threading for running the chatbot loop in a separate thread


# --- AI Chatbot Code ---
# Initialize the Detoxify model for toxicity detection.
# 'original-small' is a good balance of performance and size.
model_detoxify = Detoxify('original-small')

class ChatbotModel(nn.Module):
    """
    A simple neural network model for the chatbot.
    It consists of three fully connected layers with ReLU activations and dropout for regularization.
    """
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, 128)
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 64)
        # Third fully connected layer (output layer)
        self.fc3 = nn.Linear(64, output_size)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # No activation on the output layer for CrossEntropyLoss
        return x

class ChatbotAssistant:
    """
    Manages the chatbot's knowledge base, training, and message processing.
    """
    def __init__(self, intents_path, function_mappings=None):
        self.model = None # The PyTorch model
        self.intents_path = intents_path # Path to the intents JSON file
        self.documents = [] # List of (tokenized_pattern, tag) tuples
        self.vocabulary = [] # All unique words in the patterns
        self.intents = [] # List of unique intent tags
        self.intents_responses = {} # Dictionary mapping intent tags to their responses
        self.function_mappings = function_mappings # Optional: map intents to Python functions
        self.X = None # Input features (bag-of-words)
        self.y = None # Output labels (intent indices)

    @staticmethod
    def tokenize_and_lemmatize(text):
        """
        Tokenizes the input text and lemmatizes each word.
        Lemmatization reduces words to their base form (e.g., "running" -> "run").
        """
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        """
        Converts a list of words into a bag-of-words representation based on the chatbot's vocabulary.
        A bag-of-words is a binary vector where each element is 1 if the corresponding word
        from the vocabulary is present in the input words, and 0 otherwise.
        """
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        """
        Loads intents from the JSON file, tokenizes patterns, builds vocabulary,
        and populates documents and intent responses.
        """
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                # Add intent tag to the list of unique intents if not already present
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    # Store responses for each intent tag
                    self.intents_responses[intent['tag']] = intent['responses']

                # Process each pattern within the intent
                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words) # Add words to vocabulary
                    self.documents.append((pattern_words, intent['tag'])) # Store patterns with their tags

            # Sort and remove duplicates from the vocabulary
            self.vocabulary = sorted(set(self.vocabulary))
        else:
            raise FileNotFoundError(f"Intents file not found at: {self.intents_path}")


    def prepare_data(self):
        """
        Prepares the training data (X and y) from the parsed intents.
        X will be a list of bag-of-words vectors, and y will be corresponding intent indices.
        """
        bags = []
        indices = []
        for document in self.documents:
            words = document[0] # Tokenized pattern words
            bag = self.bag_of_words(words) # Convert words to bag-of-words vector
            intent_index = self.intents.index(document[1]) # Get numerical index of the intent
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags) # Convert to NumPy array for PyTorch
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        """
        Trains the ChatbotModel using the prepared data.
        """
        # Convert NumPy arrays to PyTorch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long) # Long type for CrossEntropyLoss labels
        dataset = TensorDataset(X_tensor, y_tensor) # Create a dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Create a data loader

        # Initialize the model, criterion (loss function), and optimizer
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # Adam optimizer

        print(f"Starting model training for {epochs} epochs...")
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad() # Clear gradients before each batch
                outputs = self.model(batch_X) # Forward pass
                loss = criterion(outputs, batch_y) # Calculate loss
                loss.backward() # Backward pass (compute gradients)
                optimizer.step() # Update model parameters
                running_loss += loss.item() # Accumulate loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")
        print("Model training complete.")

    def save_model(self, model_path, dimensions_path):
        """
        Saves the trained model's state dictionary and input/output dimensions.
        """
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)
        print(f"Model saved to {model_path} and dimensions to {dimensions_path}")

    def load_model(self, model_path, dimensions_path):
        """
        Loads a pre-trained model and its dimensions.
        """
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        # Load state dictionary, mapping to CPU if GPU is not available
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval() # Set model to evaluation mode (disables dropout, etc.)
        print(f"Model loaded from {model_path} and dimensions from {dimensions_path}")

    def process_message(self, input_message, force_last_response=False):
        """
        Processes an input message, predicts the intent, and returns a response.
        If force_last_response is True, it will always return the last response
        defined for an intent, useful for specific scenarios like high toxicity.
        """
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        # Convert bag-of-words to a PyTorch tensor
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        with torch.no_grad(): # Disable gradient calculation for inference
            predictions = self.model(bag_tensor)

        # Get the index of the highest prediction score
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        print(f"Predicted intent for '{input_message}': {predicted_intent}")

        # --- Update Tkinter popup with predicted tag ---
        if root and tag_display_var:
            # Use root.after to schedule the update on the main Tkinter thread
            root.after(0, lambda: tag_display_var.set(f"Predicted Tag: {predicted_intent}"))
        # --- End Tkinter update ---

        # Check if there's a specific function mapped to this intent
        if self.function_mappings and predicted_intent in self.function_mappings:
            # Call the mapped function. Assume it returns a string for the response.
            function_response = self.function_mappings[predicted_intent]()
            if function_response:
                return function_response # Return the response from the function
            else:
                # If function returns None, fall back to general intent responses
                pass

        # If no function mapping or function returned None, use general intent responses
        responses = self.intents_responses.get(predicted_intent)
        if responses:
            if force_last_response:
                return responses[-1] # Return the last response
            else:
                return random.choice(responses) # Return a random response
        else:
            return "I'm not sure how to respond to that." # Fallback if no responses defined

# --- Helper function for AI prediction ---
def get_stocks():
    """
    Example function to simulate getting stock information.
    This function is mapped to the 'stocks' intent.
    """
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']
    return f"Here are some popular stocks: {', '.join(random.sample(stocks, 3))}"

# Initialize the chatbot assistant globally
# This will try to load the model. If files are not found, it will print an error
# and exit, prompting the user to train the model first.
chatbot_assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
try:
    chatbot_assistant.parse_intents()
    chatbot_assistant.load_model('chatbot_model.pth', 'dimensions.json')
    print("Chatbot model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading chatbot files: {e}. Make sure 'intents.json', 'chatbot_model.pth', and 'dimensions.json' are in the correct directory.")
    print("You might need to train your model first if these files don't exist.")
    # Exiting for now, adjust based on your needs (e.g., offer to train automatically)
    exit() # Exiting for now, adjust based on your needs

def get_ai_response(user_message):
    """
    Processes a user message through the AI chatbot and returns a response.
    Includes toxicity checking using the Detoxify model.
    """
    if not user_message:
        return "I didn't receive a message."

    # Predict toxicity scores for the user message
    results = model_detoxify.predict(user_message)
    print(f"Detoxify scores for '{user_message}': {results}")

    # Check if the toxicity score exceeds a threshold
    if results['toxicity'] > 0.5:
        if results['toxicity'] >= 0.97: # High toxicity threshold
            # For highly toxic messages, force a specific, predefined response
            response = chatbot_assistant.process_message(user_message, force_last_response=True)
            # Fallback if the forced response is not found in intents
            if not response:
                response = "Please be respectful in your communication."
            return response
        else: # Moderately toxic messages
            # For moderately toxic messages, use regular chatbot processing
            response = chatbot_assistant.process_message(user_message, force_last_response=False)
            return response if response else "I'm not sure how to respond to that."
    else:
        # If not toxic, process with the regular chatbot
        response = chatbot_assistant.process_message(user_message, force_last_response=False)
        return response if response else "I'm not sure how to respond to that."

# --- WhatsApp Selenium Integration ---

def initialize_whatsapp_browser():
    """
    Initializes a Chrome browser instance for WhatsApp Web.
    Requires the user to manually scan the QR code for login.
    """
    print("Initializing Chrome browser for WhatsApp Web...")
    # Initialize Chrome WebDriver. Ensure chromedriver is in your PATH or specify its path.
    driver = webdriver.Chrome()
    try:
        print("Opening WhatsApp Web. Please scan the QR code if prompted.")
        driver.get('https://web.whatsapp.com')
        # Wait for user to manually log in by scanning QR code
        input("Press Enter in this console AFTER you have successfully logged in by scanning the QR code...")
        return driver
    except Exception as e:
        print(f"Error initializing browser or logging in: {e}")
        if driver:
            driver.quit() # Close browser if an error occurs during initialization
        return None

def find_and_click_chat(driver, contact_name):
    """
    Locates the search bar, types the contact name, and clicks on the chat.
    """
    print(f"Attempting to find and click the chat with '{contact_name}'...")
    try:
        # Find the search box element using its role and data-tab attribute
        # This XPath might need adjustment if WhatsApp Web's HTML structure changes.
        search_box_xpath = '//div[@role="textbox"][@data-tab="3"]'
        search_box = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, search_box_xpath))
        )
        search_box.send_keys(contact_name)
        time.sleep(3) # Allow search results to populate

        # Click on the chat element using the contact's name (title attribute)
        chat_xpath = f'//span[@title="{contact_name}"]'
        chat = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, chat_xpath))
        )
        chat.click()
        time.sleep(5) # Give time for the chat window to fully open and load
        print(f"Successfully opened chat with '{contact_name}'.")
        return True
    except Exception as e:
        print(f"Error finding or clicking chat with '{contact_name}': {e}")
        driver.save_screenshot("chat_find_error.png") # Save screenshot for debugging
        return False

def get_last_message(driver):
    """
    Retrieves the text of the last message in the currently open chat.
    It looks for messages with classes 'message-in' (incoming) or 'message-out' (outgoing).
    """
    print("Attempting to retrieve the last message in the chat...")
    try:
        # Wait until at least one message element is present
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.message-in, div.message-out'))
        )
        # Find all message elements
        messages = driver.find_elements(By.CSS_SELECTOR, 'div.message-in, div.message-out')
        if messages:
            last_message_element = messages[-1] # Get the last message element
            last_message_text = last_message_element.text
            print(f"Last message found: '{last_message_text}'")
            return last_message_text
        else:
            print("No messages found in this chat.")
            return None
    except Exception as e:
        print(f"Error retrieving messages: {e}")
        driver.save_screenshot("get_messages_error.png") # Save screenshot for debugging
        return None

def send_message_selenium(driver, message_text):
    """
    Sends a given message to the current chat.
    It locates the message input box, types the message, and clicks the send button.
    """
    print(f"Attempting to send message: '{message_text}'...")
    try:
        # Find the message input box
        # This XPath might need adjustment if WhatsApp Web's HTML structure changes.
        message_input_xpath = '//div[@role="textbox"][@data-tab="10"]'
        message_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, message_input_xpath))
        )
        time.sleep(1) # Small sleep to ensure the element is fully ready
        message_box.send_keys(message_text) # Type the message
        print("Message typed.")

        # Find and click the send button
        send_button_xpath = '//button[@aria-label="Send"]'
        send_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, send_button_xpath))
        )
        send_button.click() # Click the send button
        print("Send button clicked. Message should be sent.")
        time.sleep(2) # Give WhatsApp a moment to process the send action
        return True
    except Exception as e:
        print(f"Error sending message: {e}")
        driver.save_screenshot("send_message_error.png") # Save screenshot for debugging
        return False

# Global variables for Tkinter window
root = None
tag_display_var = None

def create_gui():
    """
    Creates the Tkinter window for displaying the predicted tag.
    """
    global root, tag_display_var

    root = tk.Tk()
    root.title("Chatbot Tag Display")
    root.geometry("300x100")
    root.resizable(False, False) # Prevent resizing

    tag_display_var = tk.StringVar()
    tag_display_var.set("Predicted Tag: None") # Initial text

    tag_label = tk.Label(root, textvariable=tag_display_var, font=('Arial', 14), padx=10, pady=10)
    tag_label.pack(expand=True, fill='both')

    # Handle window closing gracefully
    root.protocol("WM_DELETE_WINDOW", on_closing)

def on_closing():
    """
    Handles the closing of the Tkinter window.
    """
    print("Closing Tkinter window. The chatbot process will also terminate.")
    root.destroy()
    # You might want to add a mechanism to stop the chatbot_thread more gracefully here
    # For simplicity, we'll let the main loop exit when root.mainloop() finishes.
    os._exit(0) # Force exit the entire program

def whatsapp_chatbot_loop():
    """
    Main loop for the WhatsApp chatbot.
    It continuously monitors a specific chat, processes new messages with the AI,
    and sends back AI-generated replies.
    """
    print("--- WhatsApp Chatbot (Selenium & AI Integration) ---")
    # Prompt user for the contact name to interact with
    contact_name = input("Enter the EXACT contact name for the chatbot to interact with (case-sensitive): ").strip()

    if not contact_name:
        print("Error: Contact name cannot be empty. Exiting.")
        # If the GUI is running, update it to show an error and then close
        if root and tag_display_var:
            root.after(0, lambda: tag_display_var.set("Error: Contact name empty!"))
            root.after(3000, root.destroy) # Close after 3 seconds
        return

    driver = None
    try:
        # Initialize the browser and log in to WhatsApp Web
        driver = initialize_whatsapp_browser()
        if not driver:
            print("Failed to initialize browser. Exiting.")
            if root and tag_display_var:
                root.after(0, lambda: tag_display_var.set("Error: Browser init failed!"))
                root.after(3000, root.destroy)
            return

        # Find and open the chat with the specified contact
        if not find_and_click_chat(driver, contact_name):
            print(f"Could not open chat with '{contact_name}'. Exiting.")
            if root and tag_display_var:
                root.after(0, lambda: tag_display_var.set(f"Error: Chat with '{contact_name}' not found!"))
                root.after(3000, root.destroy)
            return

        last_processed_message = None # To avoid replying to the same message repeatedly

        print("\nChatbot is now active. Monitoring for new messages...")
        if root and tag_display_var:
            root.after(0, lambda: tag_display_var.set("Chatbot Active: Monitoring..."))

        while True:
            current_last_message = get_last_message(driver)

            # Check if there's a new message and it hasn't been processed before
            if current_last_message and current_last_message != last_processed_message:
                print(f"New message detected: '{current_last_message}'")
                # Get AI response for the new message
                ai_response = get_ai_response(current_last_message)
                print(f"AI generated response: '{ai_response}'")

                if ai_response:
                    # Send the AI-generated response
                    if send_message_selenium(driver, ai_response):
                        last_processed_message = current_last_message # Mark as processed only if sent successfully
                    else:
                        print("Failed to send AI response.")
                else:
                    print("AI did not generate a response.")
            else:
                print("No new messages or message already processed. Waiting...")

            time.sleep(10) # Wait for 10 seconds before checking for new messages again

    except KeyboardInterrupt:
        print("\nChatbot stopped by user (Ctrl+C).")
        if root and tag_display_var:
            root.after(0, lambda: tag_display_var.set("Chatbot Stopped."))
            root.after(3000, root.destroy)
    except Exception as e:
        print(f"An unexpected error occurred in the chatbot loop: {e}")
        if driver:
            driver.save_screenshot("chatbot_loop_error.png") # Save screenshot on unexpected error
        if root and tag_display_var:
            root.after(0, lambda: tag_display_var.set(f"Error: {e}"))
            root.after(3000, root.destroy)
    finally:
        if driver:
            print("Closing browser...")
            driver.quit() # Ensure the browser is closed when the script exits
        # Ensure Tkinter window is closed if the chatbot loop finishes
        if root and root.winfo_exists(): # Check if window still exists before destroying
            root.after(0, root.destroy)


if __name__ == "__main__":
    # --- Model Training Section (Uncomment and run once to train your model) ---
    # Before running this, ensure you have an 'intents.json' file in the same directory.
    # A minimal example of 'intents.json' is provided below.
    #
    # To train:
    # 1. Ensure you have 'intents.json' ready.
    # 2. Uncomment the lines below.
    # 3. Run this script once. It will create 'chatbot_model.pth' and 'dimensions.json'.
    # 4. Comment these lines back out before running the main chatbot loop,
    #    as the chatbot will then load the trained model.

    # print("Training chatbot model (if needed)...")
    # You'll need an intents.json file with patterns and responses
    # Example minimal intents.json structure:
    # {
    #   "intents": [
    #     {"tag": "greeting",
    #      "patterns": ["Hi", "Hello", "Hey", "How are you"],
    #      "responses": ["Hello!", "Hi there!", "Hey!", "I'm good, thanks!"]},
    #     {"tag": "stocks",
    #      "patterns": ["Show me stocks", "What are the stocks", "Stocks info"],
    #      "responses": ["Here are some stock updates."]},
    #     {"tag": "toxic_fallback",
    #      "patterns": [], # Patterns here are not directly used, but the tag is for force_last_response
    #      "responses": ["Please speak respectfully.", "That language is not appropriate."]}
    #   ]
    # }

    # assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
    # try:
    #     assistant.parse_intents()
    #     assistant.prepare_data()
    #     assistant.train_model(batch_size=8, lr=0.001, epochs=200) # Adjust epochs as needed
    #     assistant.save_model('chatbot_model.pth', 'dimensions.json')
    #     print("Chatbot model training and saving complete.")
    # except FileNotFoundError as e:
    #     print(f"Error: {e}. Make sure 'intents.json' exists and is correctly formatted.")
    #     exit() # Exit if intents.json is missing during training

    # --- Start the Tkinter GUI and the WhatsApp Chatbot Loop ---
    create_gui() # Create the Tkinter window

    # Start the chatbot loop in a separate thread
    chatbot_thread = threading.Thread(target=whatsapp_chatbot_loop)
    chatbot_thread.daemon = True # Allow the thread to exit when the main program exits
    chatbot_thread.start()

    # Run the Tkinter main loop (this must be in the main thread)
    root.mainloop()

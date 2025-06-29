# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:01:24 2025

@author: milos
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import os
import requests
import json
import re 

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load Model and Data ---
try:
    with open('model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("ERROR: model.pkl not found. Please ensure the model file is present.")
    model = None

try:
    df = pd.read_csv('final_preprocessed_data.csv')
except FileNotFoundError:
    print("ERROR: final.csv not found. Please ensure the data file is present.")
    df = pd.DataFrame()

# --- Gemini API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set your Gemini API key.")

MODEL_NAME = "gemini-1.5-flash-latest" 
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
API_URL_WITH_KEY = f"{API_ENDPOINT}?key={API_KEY}"


# --- Helper Function for Text Generation (Direct HTTP Request) ---
def generate_description_http(features, price=None):
    """
    Uses direct HTTP requests to the Gemini API to generate a house description.
    """
    # Create a more detailed narrative prompt for the model
    narrative_prompt = f"""
    Generate a compelling and attractive real estate listing description for a house in Iowa.
    The description should be about 4-5 sentences long and woven into a cohesive narrative.
    Incorporate the following features into the description:
    - {features}
    """

    if price:
        narrative_prompt += f"\n- Highlight that it is attractively priced at approximately ${price:,.0f}."

    narrative_prompt += "\nMake it sound appealing to potential buyers, creating a sense of home and value."

    payload = {
        "contents": [{"parts": [{"text": narrative_prompt}]}]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(API_URL_WITH_KEY, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        result = response.json()

        if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
            description = result['candidates'][0]['content']['parts'][0]['text']
            return description.strip()
        elif 'error' in result:
            return f"Error from API: {result['error'].get('message', 'Unknown error')}"
        else:
            # more detailed error for debugging
            return f"Error: Could not parse description. Full API response: {json.dumps(result)}"

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the Gemini API: {e}"
    except json.JSONDecodeError:
        return f"Error: Could not parse JSON response. Response text: {response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- HTML & Frontend ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iowa Realty Chatbot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .chat-bubble-user { background-color: #3b82f6; color: white; }
        .chat-bubble-bot { background-color: #e5e7eb; color: #1f2937; }
        .chat-scroll { scroll-behavior: smooth; }
        #chat-window::-webkit-scrollbar { width: 6px; }
        #chat-window::-webkit-scrollbar-track { background: #f1f5f9; }
        #chat-window::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px;}
        #chat-window::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        /* Style for clickable option buttons */
        .chat-option-button {
            display: block;
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            border-radius: 0.5rem;
            background-color: #ffffff;
            border: 1px solid #d1d5db;
            color: #1e40af;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s, color 0.2s;
        }
        .chat-option-button:hover {
            background-color: #eff6ff;
            color: #1d4ed8;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="w-full max-w-2xl h-full sm:h-[90vh] sm:max-h-[700px] flex flex-col bg-white shadow-2xl rounded-2xl">
        <header class="bg-blue-600 text-white p-4 rounded-t-2xl flex items-center shadow-md">
             <svg class="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z"></path></svg>
            <div>
                <h1 class="text-xl font-bold">Iowa Realty Assistant</h1>
                <p class="text-sm opacity-90">Your AI-powered guide to Iowa real estate</p>
            </div>
        </header>

        <main id="chat-window" class="flex-1 p-6 overflow-y-auto chat-scroll bg-gray-50">
            <div class="flex justify-start mb-4">
                <div class="chat-bubble-bot max-w-lg p-3 rounded-lg shadow">
                    <p class="text-sm">Hello! I'm your real estate assistant. How can I help you today? You can ask me to:</p>
                    <ul class="list-disc list-inside mt-2 text-sm space-y-1">
                        <li>Estimate a price by typing: <br><b>price for [Total SF] sf built in [Year]</b></li>
                        <li>Create a description by typing: <br><b>description for [features]</b></li>
                    </ul>
                </div>
            </div>
        </main>

        <div id="typing-indicator" class="px-6 pb-2 hidden">
             <div class="flex justify-start">
                 <div class="chat-bubble-bot p-3 rounded-lg shadow">
                     <div class="flex items-center space-x-1">
                         <span class="text-sm text-gray-500">Typing</span>
                         <div class="w-1 h-1 bg-gray-500 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                         <div class="w-1 h-1 bg-gray-500 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                         <div class="w-1 h-1 bg-gray-500 rounded-full animate-pulse"></div>
                     </div>
                 </div>
             </div>
        </div>

        <footer class="p-4 bg-white border-t rounded-b-2xl">
            <form id="chat-form" class="flex items-center space-x-3">
                <input type="text" id="message-input" placeholder="e.g., price for 1800 sf built in 2005" class="flex-1 w-full px-4 py-2 text-sm bg-gray-100 border border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition">
                <button type="submit" class="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-300 transform hover:scale-110">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clip-rule="evenodd" fill-rule="evenodd"></path></svg>
                </button>
            </form>
        </footer>
    </div>

    <script>
        const chatWindow = document.getElementById('chat-window');
        const chatForm = document.getElementById('chat-form');
        const messageInput = document.getElementById('message-input');
        const typingIndicator = document.getElementById('typing-indicator');

        // Function to display messages. It now handles both text and HTML.
        function displayMessage(message, sender, isHtml = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex mb-4 ${sender === 'user' ? 'justify-end' : 'justify-start'}`;
            const bubble = document.createElement('div');
            bubble.className = `max-w-lg p-3 rounded-lg shadow ${sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot'}`;

            if (isHtml) {
                bubble.innerHTML = message; // Use innerHTML for rich content
            } else {
                bubble.innerText = message; // Use innerText for plain text to prevent XSS
            }

            messageDiv.appendChild(bubble);
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        // Central function to send data to the backend
        async function sendRequest(payload) {
            typingIndicator.classList.remove('hidden');
            chatWindow.scrollTop = chatWindow.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();

                // Display the primary text response
                if (data.response) {
                    displayMessage(data.response, 'bot');
                }

                // If there's a follow-up with interactive options, display it
                if (data.follow_up_html) {
                    displayMessage(data.follow_up_html, 'bot', true);
                }

            } catch (error) {
                console.error('Error:', error);
                displayMessage('Sorry, something went wrong. Please check the console for details.', 'bot');
            } finally {
                typingIndicator.classList.add('hidden');
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        }

        // Handle form submission for text input
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const message = messageInput.value.trim();
            if (!message) return;

            displayMessage(message, 'user');
            messageInput.value = '';

            sendRequest({ message: message });
        });

        // Handle clicks on dynamically added option buttons
        chatWindow.addEventListener('click', (e) => {
            const target = e.target.closest('.chat-option-button');
            if (!target) return;

            const action = target.dataset.action;
            const context = JSON.parse(target.dataset.context || '{}');
            const displayText = target.innerText;

            // Display the user's choice as if they typed it
            displayMessage(displayText, 'user');

            sendRequest({
                message: action, // Send the action keyword
                context: context // Send the stored context back to the server
            });
        });
    </script>
</body>
</html>
"""


# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main chat page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    """Handles the main chatbot logic."""
    if model is None or df.empty:
        return jsonify({'response': 'Error: Backend model or data not loaded.'}), 500

    data = request.json
    user_message = data.get('message', '').lower()
    context = data.get('context', {}) # Get context from the frontend
    response_text = "Sorry, I don't understand. Please ask for a 'price for [Total SF] sf built in [Year]' or a 'description for [features]'."
    follow_up_html = None

    price_pattern = re.compile(r"price for\s*([\d,]+)\s*sf\s+built in\s*(\d{4})")
    match = price_pattern.search(user_message)

    if match:
        try:
            total_sf_str = match.group(1).replace(',', '')
            total_sf = int(total_sf_str)
            year_built = int(match.group(2))

            model_features = [col for col in df.columns if col != 'SalePrice']
            synthetic_data = {}
            for feature in model_features:
                if feature == 'Total SF': synthetic_data[feature] = total_sf
                elif feature == 'Year Built': synthetic_data[feature] = year_built
                elif pd.api.types.is_numeric_dtype(df[feature]): synthetic_data[feature] = df[feature].median()
                else: synthetic_data[feature] = df[feature].mode()[0]

            features_df = pd.DataFrame([synthetic_data])[model_features]
            predicted_price = predicted_price = float(np.expm1(model.predict(features_df)[0]))

            response_text = f"The estimated price for a {total_sf:,} sq ft house built in {year_built} is ${predicted_price:,.2f}. What would you like to do next?"

            new_context = {
                'total_sf': total_sf,
                'year_built': year_built,
                'predicted_price': predicted_price
            }
            context_json = json.dumps(new_context)

            follow_up_html = f"""
                <button class="chat-option-button" data-action="describe_with_price" data-context='{context_json}'>
                    Create description with this price
                </button>
                <button class="chat-option-button" data-action="describe_without_price" data-context='{context_json}'>
                    Create description without a price
                </button>
                <button class="chat-option-button" data-action="list_features" data-context='{context_json}'>
                    List features for a more accurate price
                </button>
            """
        except Exception as e:
            response_text = f"An error occurred during prediction: {e}"

    elif user_message.startswith('description for'):
        features = user_message.replace('description for', '').strip()
        if len(features) < 10:
            response_text = "Please provide more detailed features (e.g., 'a 3 bedroom, 2 bathroom house with a modern kitchen')."
        else:
            response_text = generate_description_http(features)

    elif user_message == 'describe_with_price':
        price = context.get('predicted_price')
        features = f"{context.get('total_sf'):,} sq ft, built in {context.get('year_built')}"
        response_text = generate_description_http(features, price)

    elif user_message == 'describe_without_price':
        features = f"{context.get('total_sf'):,} sq ft, built in {context.get('year_built')}"
        response_text = generate_description_http(features) # Price is omitted

    elif user_message == 'list_features':
        all_features = [col for col in df.columns if col != 'SalePrice' and col not in ['Total SF', 'Year Built']]
        display_features = all_features[:5]
        response_text = (
            "To get a more accurate price, you can include features like: "
            f"{', '.join(display_features)}. \n\nPlease rephrase your request, for example: "
            "'price for 1800 sf built in 2005 with 3 bedrooms and 2 garage cars'."
        )


    return jsonify({'response': response_text, 'follow_up_html': follow_up_html})


if __name__ == '__main__':
    if API_KEY == "YOUR_GOOGLE_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: API_KEY is not set. Please replace the         !!!")
        print("!!! placeholder with your actual Google API key.            !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    app.run(debug=True)
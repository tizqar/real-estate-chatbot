What Can It Do?
Estimate House Prices: Give it the size and age of a house in Iowa, and it will give you a good idea of what it might be worth.

Write Property Descriptions: Tell it about a property (e.g., "a cozy 3-bedroom house with a big backyard"), and it will write a creative, appealing description for you using Google's Gemini AI.

Chat Interactively: After you get a price, the chatbot will ask if you want it to write a description based on that estimate, making the conversation feel more natural.

Behind the Scenes:

For Price Estimates: When you ask for a price, the app uses a smart program (an XGBoost machine learning model) that has been trained on lots of real housing data from Iowa. It looks at the details you provide and makes an educated guess.

For Descriptions: When you ask for a description, the app sends your request over to Google's Gemini AI. Gemini then works its magic and writes a nice, human-sounding description, which the app shows back to you.

What's Under the Hood?
Backend: The main engine is built with Python and Flask.

Machine Learning:Scikit-learn, XGBoost, and Pandas to build and train price prediction model.

Creative AI: Google's Gemini API is used for writing the property descriptions.

Frontend: The chat interface is built with simple HTML, JavaScript, and styled with Tailwind CSS to make it look clean and modern.

Getting It Running: Setup Guide
Want to run this on your own computer? Just follow these steps.

Set Up a Virtual Environment:
This is like a private workspace for the project so it doesn't mess with other Python stuff on your computer.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the Goodies:
This command installs all the tools the project needs to run.

pip install -r requirements.txt

Add Your API Key:
The app needs a key to talk to Google's Gemini AI. You'll need to set this up as an environment variable.

# On Linux/macOS
export GEMINI_API_KEY="your_actual_api_key"

# On Windows
set GEMINI_API_KEY="your_actual_api_key"

Check for Important Files:
Before you run it, make sure the trained model (model1.pkl) and the data file (final_preprocessed_data.csv) are in the project folder. Running the model_and_preprocessing.py script will create these for you.

Launch It!

python app.py

Now, you can open your web browser and go to http://127.0.0.1:5000 to start chatting!
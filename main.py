import openai
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "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"

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    
    # Generate AI response using GPT-3
    response = openai.Completion.create(
        engine="davinci",
        prompt=user_input,
        max_tokens=50  # Set the desired response length
    )
    
    ai_response = response
    
    return jsonify({'ai_response': ai_response})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)

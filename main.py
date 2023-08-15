from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# Define your chatbot model architecture (simplified example)
class ChatbotModel(nn.Module):
    def __init__(self):
        super(ChatbotModel, self).__init__()
        # Define your model layers here

    def forward(self, input_text):
        # Implement the model's forward pass to generate a response
        return generated_response

# Load your pre-trained chatbot model
model = ChatbotModel()
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    
    # Process user input and generate chatbot response using the model
    generated_response = generate_response(user_input)
    
    return jsonify({'response': generated_response})

def generate_response(input_text):
    # Convert input text to a format suitable for your model
    # Use your model to generate a response based on the input
    
    # For this example, a simple echo bot behavior is shown
    response = "You said: " + input_text
    return response

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)

from flask import Flask, render_template, request, jsonify 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import CORS

import os
#ngrok http --url=classic-proven-kingfish.ngrok-free.app 80
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Set up CORS for regular HTTP routes
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

# Set up SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create a custom AttentionLayer exactly as in your original code
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix to learn the importance of each feature
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        # Bias term
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        # Context vector to compare with the transformed features
        self.u = self.add_weight(name="context_vector",
                                 shape=(input_shape[-1],),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        att_scores = tf.tensordot(u_it, self.u, axes=1)
        att_weights = tf.nn.softmax(att_scores, axis=1)
        weighted_output = tf.reduce_sum(x * tf.expand_dims(att_weights, -1), axis=1)
        return weighted_output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# Load your actions array
actions = np.array([
    'a', 'again', 'april', 'are_you_deaf', 'assignment', 'august', 'auntie',
    'b', 'baby', 'beer', 'best_friend', 'black', 'blind', 'blue', 'boy',
    'boyfriend', 'bread', 'brother', 'brown', 'c', 'cat', 'chicken', 'child',
    'childhood', 'close_friend', 'coffee', 'cold', 'computer',
    'correct', 'cousin', 'crab', 'd',
    'dark', 'daughter', 'deaf', 'deaf_blind', 'december', 'do_you_like', 'dog',
    'dont_know', 'dont_understand',  'e', 'egg', 'eight',
    'English', 'excuse_me', 'f', 'family', 'fast', 'father', 'february',
    'Filipino', 'fish', 'five', 'four', 'friday', 'friend', 'g', 'girl',
    'good_afternoon', 'good_evening', 'good_morning', 'goodbye',
    'grandfather', 'grandmother', 'gray', 'green', 'h', 'handsome', 'happy_birthday',
    'hard_of_hearing', 'hearing', 'hello', 'hot', 'how', 'how_are_you',
    'how_much', 'hungry', 'i', 'i_forget', 'i_know_a_little_sign',
    'i_live_in', 'i_love_you', 'i_miss_you', 'im_fine', 'im_not_fine',
    'is_she_your_mother', 'j', 'january', 'juice', 'july', 'june', 'k',
    'keep_working', 'know', 'l', 'library', 'light', 'longanisa', 'm', 'man',
    'march', 'married', 'mathematics', 'may', 'maybe', 'meat', 'milk', 'monday',
    'mother', 'my_name_is', 'n', 'nice_to_meet_you',
    'nice_to_meet_you_too', 'nine', 'no', 'no_sugar', 'november',
    'o', 'october', 'one', 'orange', 'p', 'paper', 'parents', 'pink',
    'please', 'please_come_here', 'please_sign_slowly', 'principal',
    'q', 'r', 'red', 'rice', 's', 'saturday', 'school', 'science',
    'see_you_later', 'see_you_tomorrow', 'september', 'seven', 'shrimp',
    'sign_language', 'sister', 'six', 'slow', 'son', 'sorry', 'spaghetti',
    'step_brother', 'step_sister', 'sugar', 'sunday', 't',
    'take_care', 'tea', 'teacher', 'ten', 'thank_you',
    'they_are_kind', 'they_are_pretty', 'three', 'thursday', 'today',
    'tomorrow', 'tuesday', 'two', 'u', 'uncle', 'understand',
    'v', 'violet', 'w', 'wait', 'wednesday', 'what',
    'what_is_your_name', "whats_your_favorite_subject", 'wheelchair_person',
    'where', 'which', 'white', 'who', 'why', 'wine', 'woman',
    'wrong', 'x', 'y', 'yellow', 'yes', 'yesterday', 'you_sign_fast', 'youre_welcome',
])

# Load model
model = None

def load_lstm_model():
    global model
    model = load_model('LSTMdeepattention_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

# Keep the HTTP endpoint for compatibility
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get keypoints data from the request
        keypoints_data = request.json.get('keypoints', [])
        
        # Process data and make prediction
        result = process_prediction(keypoints_data)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)

# Process prediction data and return results
def process_prediction(keypoints_data):
    try:
        # Convert to numpy array
        sequence = np.array(keypoints_data)
        
        # Ensure we have the right shape for prediction
        if sequence.shape[0] != 120:
            return {'error': 'Need exactly 120 frames of keypoints'}
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)
        res = model.predict(sequence_batch)[0]
        
        # Get the predicted action
        predicted_action = actions[np.argmax(res)]
        confidence = float(res[np.argmax(res)])
        
        # Get top predictions
        top_indices = np.argsort(res)[::-1][:5]  # Get top 5 predictions
        top_actions = [actions[i] for i in top_indices]
        top_probabilities = [float(res[i]) for i in top_indices]
        
        return {
            'prediction': predicted_action,
            'confidence': confidence,
            'top_predictions': [{'action': action, 'probability': prob} 
                              for action, prob in zip(top_actions, top_probabilities)]
        }
    except Exception as e:
        return {'error': str(e)}

# WebSocket endpoint for predictions
@socketio.on('predict_sign')
def handle_prediction(data):
    try:
        # Get keypoints data from the WebSocket message
        keypoints_data = data.get('keypoints', [])
        
        # Process prediction
        result = process_prediction(keypoints_data)
        
        if 'error' in result:
            emit('prediction_error', {'error': result['error']})
            return
        
        # Send prediction back to all clients (including the sender)
        emit('prediction_result', result, broadcast=True)
        
    except Exception as e:
        emit('prediction_error', {'error': str(e)})

# WebSocket connection event
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to server'})

# WebSocket disconnection event
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize model
    load_lstm_model()
    
    # Run Flask app with SocketIO - bind to all interfaces
    # Make sure to use 0.0.0.0 to accept connections from any IP
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Layer
# import base64
# import os
# import random

# # Custom Attention Layer definition
# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Weight matrix to learn the importance of each feature
#         self.W = self.add_weight(name="att_weight",
#                                  shape=(input_shape[-1], input_shape[-1]),
#                                  initializer="glorot_uniform",
#                                  trainable=True)
#         # Bias term
#         self.b = self.add_weight(name="att_bias",
#                                  shape=(input_shape[-1],),
#                                  initializer="zeros",
#                                  trainable=True)
#         # Context vector to compare with the transformed features
#         self.u = self.add_weight(name="context_vector",
#                                  shape=(input_shape[-1],),
#                                  initializer="glorot_uniform",
#                                  trainable=True)
#         super(AttentionLayer, self).build(input_shape)

#     def call(self, x):
#         u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
#         att_scores = tf.tensordot(u_it, self.u, axes=1)
#         att_weights = tf.nn.softmax(att_scores, axis=1)
#         weighted_output = tf.reduce_sum(x * tf.expand_dims(att_weights, -1), axis=1)
#         return weighted_output
    
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[2])

# # Initialize Flask application
# app = Flask(__name__)

# # MediaPipe setup
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # Load the sign language recognition model
# model = load_model('LSTMdeepattention_model.h5', custom_objects={'AttentionLayer': AttentionLayer})

# # Define sign language actions/classes
# actions = np.array([
#     'a', 'again', 'april', 'are_you_deaf', 'assignment', 'august', 'auntie',
#     'b', 'baby', 'beer', 'best_friend', 'black', 'blind', 'blue', 'boy',
#     'boyfriend', 'bread', 'brother', 'brown', 'c', 'cat', 'chicken', 'child',
#     'childhood', 'close_friend', 'coffee', 'cold', 'computer',
#     'correct', 'cousin', 'crab', 'd',
#     'dark', 'daughter', 'deaf', 'deaf_blind', 'december', 'do_you_like', 'dog',
#     'dont_know', 'dont_understand',  'e', 'egg', 'eight',
#     'English', 'excuse_me', 'f', 'family', 'fast', 'father', 'february',
#     'Filipino', 'fish', 'five', 'four', 'friday', 'friend', 'g', 'girl',
#     'good_afternoon', 'good_evening', 'good_morning', 'goodbye',
#     'grandfather', 'grandmother', 'gray', 'green', 'h', 'handsome', 'happy_birthday',
#     'hard_of_hearing', 'hearing', 'hello', 'hot', 'how', 'how_are_you',
#     'how_much', 'hungry', 'i', 'i_forget', 'i_know_a_little_sign',
#     'i_live_in', 'i_love_you', 'i_miss_you', 'im_fine', 'im_not_fine',
#     'is_she_your_mother', 'j', 'january', 'juice', 'july', 'june', 'k',
#     'keep_working', 'know', 'l', 'library', 'light', 'longanisa', 'm', 'man',
#     'march', 'married', 'mathematics', 'may', 'maybe', 'meat', 'milk', 'monday',
#     'mother', 'my_name_is', 'n', 'nice_to_meet_you',
#     'nice_to_meet_you_too', 'nine', 'no', 'no_sugar', 'november',
#     'o', 'october', 'one', 'orange', 'p', 'paper', 'parents', 'pink',
#     'please', 'please_come_here', 'please_sign_slowly', 'principal',
#     'q', 'r', 'red', 'rice', 's', 'saturday', 'school', 'science',
#     'see_you_later', 'see_you_tomorrow', 'september', 'seven', 'shrimp',
#     'sign_language', 'sister', 'six', 'slow', 'son', 'sorry', 'spaghetti',
#     'step_brother', 'step_sister', 'sugar', 'sunday', 't',
#     'take_care', 'tea', 'teacher', 'ten', 'thank_you',
#     'they_are_kind', 'they_are_pretty', 'three', 'thursday', 'today',
#     'tomorrow', 'tuesday', 'two', 'u', 'uncle', 'understand',
#     'v', 'violet', 'w', 'wait', 'wednesday', 'what',
#     'what_is_your_name', "whats_your_favorite_subject", 'wheelchair_person',
#     'where', 'which', 'white', 'who', 'why', 'wine', 'woman',
#     'wrong', 'x', 'y', 'yellow', 'yes', 'yesterday', 'you_sign_fast', 'youre_welcome',
# ])

# # Generate distinct colors for visualization
# def generate_colors(num_colors):
#     colors = []
#     for i in range(num_colors):
#         hue = int(179 * i / num_colors)
#         saturation = random.randint(150, 255)
#         value = random.randint(150, 255)
        
#         hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
#         bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
#         colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    
#     return colors

# colors = generate_colors(len(actions))

# # MediaPipe utility functions
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def draw_styled_landmarks(image, results):
#     # Draw pose connections
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
#                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                                 ) 
#     # Draw left hand connections
#     if results.left_hand_landmarks:
#         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
#                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                                 ) 
#     # Draw right hand connections  
#     if results.right_hand_landmarks:
#         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                 )

# # Function to extract keypoints from MediaPipe results
# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, lh, rh])

# # Function to visualize prediction probabilities
# def prob_viz(res, actions):
#     # Return top 5 predictions
#     sorted_indices = np.argsort(res)[::-1][:5]  # Get indices of top 5 predictions
    
#     top_predictions = []
#     for idx in sorted_indices:
#         if res[idx] > 0.01:  # Only include predictions with probability > 1%
#             top_predictions.append({
#                 'action': actions[idx],
#                 'probability': float(res[idx])
#             })
    
#     return top_predictions

# # Global variables for prediction
# sequence_length = 120
# sequence = []
# sentence = []
# predictions = []
# threshold = 0.5

# # Main route to serve the web interface
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Process frames sent from the client
# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     global sequence, sentence, predictions
    
#     try:
#         # Get image data from request
#         data = request.json
#         image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64, part
#         show_landmarks = data.get('showLandmarks', True)  # Get landmark preference, default to True
        
#         # Decode base64 image
#         image_bytes = base64.b64decode(image_data)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             return jsonify({'error': 'Invalid image data'}), 400
        
#         # Initialize MediaPipe Holistic model for this frame
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             # Make detections
#             image, results = mediapipe_detection(frame, holistic)
            
#             # Create a copy for landmark visualization
#             display_image = image.copy()
            
#             # Draw landmarks on the image if requested
#             if show_landmarks:
#                 draw_styled_landmarks(display_image, results)
            
#             # Extract keypoints for prediction (landmarks are always extracted regardless of display preference)
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
            
#             # Keep only the last 120 frames
#             if len(sequence) > sequence_length:
#                 sequence = sequence[-sequence_length:]
            
#             # Convert the processed image back to base64 to send to the client
#             _, buffer = cv2.imencode('.jpg', display_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
#             processed_image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            
#             # Process the results
#             result_data = {
#                 'sentence': ' '.join(sentence) if sentence else '',
#                 'top_predictions': [],
#                 'landmarks_detected': {
#                     'pose': results.pose_landmarks is not None,
#                     'left_hand': results.left_hand_landmarks is not None,
#                     'right_hand': results.right_hand_landmarks is not None
#                 },
#                 'processed_image': processed_image  # Send the processed image (with or without landmarks)
#             }
            
#             # Make prediction if we have enough frames
#             if len(sequence) == sequence_length:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                 predictions.append(np.argmax(res))
                
#                 # Get top predictions for visualization
#                 result_data['top_predictions'] = prob_viz(res, actions)
                
#                 # Update sentence based on predictions
#                 if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
#                     if res[np.argmax(res)] > threshold:
#                         if len(sentence) > 0:
#                             if actions[np.argmax(res)] != sentence[-1]:
#                                 sentence.append(actions[np.argmax(res)])
#                         else:
#                             sentence.append(actions[np.argmax(res)])
                
#                 # Limit sentence length
#                 if len(sentence) > 5:
#                     sentence = sentence[-5:]
                
#                 result_data['sentence'] = ' '.join(sentence)
            
#             return jsonify(result_data)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Route to reset the system
# @app.route('/reset', methods=['POST'])
# def reset():
#     global sequence, sentence, predictions
#     sequence = []
#     sentence = []
#     predictions = []
#     return jsonify({'status': 'reset successful'})

# if __name__ == '__main__':
    
#     # Start the Flask app
#     app.run(host='0.0.0.0', port=5000, debug=True)
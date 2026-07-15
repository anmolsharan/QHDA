from flask import Flask, render_template, request, send_file , jsonify
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import Model
from PIL import Image
import uuid
import imageio
import base64

# -------------------- Flask Setup --------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static/results")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------- Load Model --------------------

MODEL_PATH = os.path.join(BASE_DIR, "grad_best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


# -------------------- Image Preprocessing --------------------

def preprocess_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_last_conv_layer(model):
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    return conv_layers[-1] if conv_layers else None


# -------------------- GradCAM --------------------

def generate_gradcam(model, img_array, pred_index):

    last_conv_layer_name = get_last_conv_layer(model)

    if last_conv_layer_name is None:
        return None

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    conv_outputs = conv_outputs * pooled_grads

    heatmap = tf.reduce_sum(conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return None

    heatmap /= max_val

    return heatmap.numpy()


# -------------------- ScoreCAM --------------------

def generate_scorecam(model, img_array, pred_index, top_k=50):

    last_conv_layer_name = get_last_conv_layer(model)

    if not last_conv_layer_name:
        return None

    fmap_model = Model(inputs=model.input, outputs=model.get_layer(last_conv_layer_name).output)

    fmap = fmap_model(img_array)[0].numpy()

    h, w, c = fmap.shape

    norms = np.linalg.norm(fmap.reshape(-1, c), axis=0)

    idxs = np.argsort(norms)[::-1]

    if top_k is not None and top_k < len(idxs):
        idxs = idxs[:top_k]

    base_img = img_array[0]

    scores = []
    maps = []

    for idx in idxs:

        act_map = fmap[:, :, idx]

        act_map_resized = cv2.resize(act_map, (base_img.shape[1], base_img.shape[0]))

        act_map_resized = act_map_resized - act_map_resized.min()

        if act_map_resized.max() != 0:
            act_map_resized = act_map_resized / act_map_resized.max()

        mask = np.expand_dims(act_map_resized, axis=-1)

        masked_input = base_img * mask

        out = model.predict(np.expand_dims(masked_input, axis=0))

        score = out[0, pred_index]

        scores.append(score)

        maps.append(act_map)

    scores = np.array(scores)

    maps = np.stack(maps, axis=-1)

    weighted_sum = np.sum(maps * scores[np.newaxis, np.newaxis, :], axis=-1)

    heatmap = np.maximum(weighted_sum, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap


# -------------------- LayerCAM --------------------

def generate_layercam(model, img_array, pred_index):

    last_conv_layer_name = get_last_conv_layer(model)

    if not last_conv_layer_name:
        return None

    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array, training=False)

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    grads_np = grads[0].numpy()

    activations = conv_outputs[0].numpy()

    elementwise = np.maximum(grads_np * activations, 0)

    heatmap = np.sum(elementwise, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap


# -------------------- Overlay Heatmap --------------------

def overlay_heatmap(img_path, heatmap, alpha=0.6):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (256, 256))

    heatmap = cv2.resize(heatmap, (256, 256))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)


# -------------------- GIF Generator --------------------

def create_gif(original_img_path, heatmap_img_path, gif_path, num_frames=20, duration=120):

    original_img = Image.open(original_img_path).convert("RGBA")

    heatmap_img = Image.open(heatmap_img_path).convert("RGBA")

    if original_img.size != heatmap_img.size:
        heatmap_img = heatmap_img.resize(original_img.size)

    frames = []

    for alpha in np.linspace(0, 1, num_frames):
        blended = Image.blend(original_img, heatmap_img, alpha)
        frames.append(blended)

    for alpha in np.linspace(1, 0, num_frames):
        blended = Image.blend(original_img, heatmap_img, alpha)
        frames.append(blended)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration,
        loop=0
    )


# -------------------- Utilities --------------------

def encode_file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


# -------------------- Routes --------------------

@app.route('/')
def home():
    return render_template('index.html')


# -------- JSON GradCAM --------

@app.route('/MLDL', methods=['POST'])
def predict_gradcam_json():

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']

    filename = str(uuid.uuid4()) + ".jpg"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    image_url = f"/static/uploads/{filename}"

    img_array = preprocess_image(filepath)

    predictions = model.predict(img_array)

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    pred_index = int(np.argmax(predictions[0]))

    confidence = float(np.round(predictions[0][pred_index] * 100, 2))

    pred_class = class_names[pred_index]

    heatmap_b64 = None
    gif_b64 = None

    if pred_class != "notumor":

        heatmap = generate_gradcam(model, img_array, pred_index)

        if heatmap is not None:

            result_img = overlay_heatmap(filepath, heatmap)

            result_path = os.path.join(RESULT_FOLDER, filename)

            cv2.imwrite(result_path, result_img)

            gif_filename = filename.replace('.jpg', '.gif')

            gif_path = os.path.join(RESULT_FOLDER, gif_filename)

            create_gif(filepath, result_path, gif_path)

            heatmap_b64 = encode_file_to_base64(result_path)

            gif_b64 = encode_file_to_base64(gif_path)

    return jsonify({
        'prediction': pred_class,
        'confidence': confidence,
        'image_url': image_url,
        'heatmap_base64': heatmap_b64,
        'gif_base64': gif_b64
    })


# -------- ScoreCAM JSON --------

@app.route('/MLDL/scorecam', methods=['POST'])
def predict_scorecam_json():

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']

    filename = str(uuid.uuid4()) + ".jpg"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    image_url = f"/static/uploads/{filename}"

    img_array = preprocess_image(filepath)

    predictions = model.predict(img_array)

    pred_index = int(np.argmax(predictions[0]))

    confidence = float(np.round(predictions[0][pred_index] * 100, 2))

    pred_class = class_names[pred_index]

    heatmap = generate_scorecam(model, img_array, pred_index)

    if heatmap is None:
        heatmap = generate_gradcam(model, img_array, pred_index)

    result_img = overlay_heatmap(filepath, heatmap)

    result_path = os.path.join(RESULT_FOLDER, filename)

    cv2.imwrite(result_path, result_img)

    gif_filename = filename.replace('.jpg', '.gif')

    gif_path = os.path.join(RESULT_FOLDER, gif_filename)

    create_gif(filepath, result_path, gif_path)

    return jsonify({
        'prediction': pred_class,
        'confidence': confidence,
        'heatmap_base64': encode_file_to_base64(result_path),
        'gif_base64': encode_file_to_base64(gif_path)
    })


# -------- LayerCAM JSON --------

@app.route('/MLDL/layercam', methods=['POST'])
def predict_layercam_json():

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']

    filename = str(uuid.uuid4()) + ".jpg"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    image_url = f"/static/uploads/{filename}"

    img_array = preprocess_image(filepath)

    predictions = model.predict(img_array)

    pred_index = int(np.argmax(predictions[0]))

    confidence = float(np.round(predictions[0][pred_index] * 100, 2))

    pred_class = class_names[pred_index]

    heatmap = generate_layercam(model, img_array, pred_index)

    if heatmap is None:
        heatmap = generate_gradcam(model, img_array, pred_index)

    result_img = overlay_heatmap(filepath, heatmap)

    result_path = os.path.join(RESULT_FOLDER, filename)

    cv2.imwrite(result_path, result_img)

    gif_filename = filename.replace('.jpg', '.gif')

    gif_path = os.path.join(RESULT_FOLDER, gif_filename)

    create_gif(filepath, result_path, gif_path)

    return jsonify({
        'prediction': pred_class,
        'confidence': confidence,
        'heatmap_base64': encode_file_to_base64(result_path),
        'gif_base64': encode_file_to_base64(gif_path)
    })


# -------------------- Download --------------------

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)


# -------------------- Run Server --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)


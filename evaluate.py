import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH      = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
TEST_DIR        = os.path.join("dataset", "test")
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32

# ── Load class names ──
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
else:
    class_names = None

# ── Load model ──
if not os.path.exists(MODEL_PATH):
    print(f" Model not found at '{MODEL_PATH}'. Run train.py first.")
    exit()

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded!\n")

# ── Test dataset (no augmentation, no shuffle) ──
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

if class_names is None:
    class_names = test_ds.class_names

print(f"Classes: {class_names}\n")

# Rescale (must match training)
rescale = tf.keras.layers.Rescaling(1.0 / 255)
test_ds = test_ds.map(lambda x, y: (rescale(x), y),
                      num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ── Predict ──
print("Running predictions on test set...")
y_true_batches = []
y_pred_batches = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred_batches.append(np.argmax(preds, axis=1))
    y_true_batches.append(np.argmax(labels.numpy(), axis=1))

y_pred = np.concatenate(y_pred_batches)
y_true = np.concatenate(y_true_batches)

# ── Metrics ──
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Confusion matrix ──
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=class_names, yticklabels=class_names,
)
plt.title("Confusion Matrix — Plant Disease Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")
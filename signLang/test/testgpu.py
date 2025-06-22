import os
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------------- ✅ GPU SETUP -------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU(s) Available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[WARNING] No GPU found. Running on CPU.")

# ---------------------- ✅ DATA PREP -------------------------
actions = np.array([
    'hello', 'my', 'name'
])

label_map = {label: num for num, label in enumerate(actions)}
DATA_PATH = os.path.join('MP_Data')
no_sequences = 30
sequence_length = 30

sequences, labels = [], []

def load_sequence(action, sequence):
    window = []
    for frame_num in range(sequence_length):
        res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
        window.append(res)
    return window, label_map[action]

print("[INFO] Loading data sequences...")

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(load_sequence, action, seq)
               for action in actions for seq in range(no_sequences)]
    for future in futures:
        window, label = future.result()
        sequences.append(window)
        labels.append(label)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("[INFO] Data loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print("y_test shape:", y_test.shape)

# ---------------------- ✅ MODEL -------------------------
model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=(30, 1662)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(64, return_sequences=False, activation='tanh')))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Model Summary:")
model.summary()

# ---------------------- ✅ CALLBACKS -------------------------
checkpoint = ModelCheckpoint('best_model_withGPU_test.keras', monitor='val_categorical_accuracy',
                             save_best_only=True, mode='max', verbose=1)

earlystop = EarlyStopping(monitor='val_categorical_accuracy', patience=20, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-6, verbose=1)

# ---------------------- ✅ TRAIN -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=32,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# ---------------------- ✅ SAVE FINAL -------------------------
model.save('final_sign_model_withGPU.keras')
print("✅ Final model saved as final_sign_model_withGPU.keras")

    import os
    import numpy as np
    import cv2
    import logging
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path ke folder dataset
    data_path = "C:/Users/Hp/Downloads/archiveemotion/train"
    emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def load_data(data_path):
        images = []
        labels = []
        for label in emotion_labels:
            label_path = os.path.join(data_path, label)
            logger.info(f"Processing label: {label}")
            for file in os.listdir(label_path):
                if file.endswith(".png"):
                    img_path = os.path.join(label_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(emotion_labels.index(label))
        logger.info(f"Loaded {len(images)} images.")
        return np.array(images), np.array(labels)

    logger.info("Starting data loading...")
    X, y = load_data(data_path)
    X = X.reshape(X.shape[0], 48, 48, 1) / 255.0  # Normalisasi
    y = np.eye(len(emotion_labels))[y]  # One-hot encoding

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    logger.info("Starting model training...")
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=64)

    # Save model
    model.save('emotion_detection_model.h5')
    logger.info("Model saved successfully.")

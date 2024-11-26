import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose

def create_improved_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        
        Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Changed to sigmoid activation
    ])
    return model

def train_and_test():
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    selected_digits = [2, 5, 9]
    train_mask = np.isin(y_train, selected_digits)
    test_mask = np.isin(y_test, selected_digits)
    
    x_train = x_train[train_mask]
    x_test = x_test[test_mask]
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    noise_factor = 0.25  # Reduced noise factor
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    print("Creating improved model...")
    model = create_improved_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse',
                 metrics=['mae'])
    
    print("Training model...")
    history = model.fit(x_train_noisy, x_train,
                       batch_size=128,
                       epochs=10,  # Increased epochs
                       validation_split=0.2,
                       verbose=1)
    
    model.save('improved_denoising_autoencoder.h5')
    
    print("Testing model...")
    test_indices = np.random.choice(len(x_test), 5)
    test_images = x_test[test_indices]
    test_noisy = x_test_noisy[test_indices]
    denoised_images = model.predict(test_noisy)
    
    mse_noisy = np.mean((test_images - test_noisy) ** 2)
    mse_denoised = np.mean((test_images - denoised_images) ** 2)
    
    plt.figure(figsize=(15, 5))
    for i in range(5):
        # Original
        plt.subplot(3, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title('Original')
        plt.axis('off')
        
        # Noisy
        plt.subplot(3, 5, i + 6)
        plt.imshow(test_noisy[i].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title('Noisy')
        plt.axis('off')
        
        # Denoised
        plt.subplot(3, 5, i + 11)
        plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title('Denoised')
        plt.axis('off')
    
    plt.savefig('improved_denoising_results.png')
    plt.close()
    
    print("\nQuality Metrics:")
    print(f"Mean Squared Error (Noisy): {mse_noisy:.4f}")
    print(f"Mean Squared Error (Denoised): {mse_denoised:.4f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('improved_training_history.png')
    plt.close()

if __name__ == "__main__":
    train_and_test()
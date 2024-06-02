from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def cnn(input_shape):

    # Define CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape+(1,)), # Just one channel
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2),

        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    # Model compilation
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    pass
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image

# Setting up directory
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(base_dir, 'data')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')

# Generate Images using ImageDataGenerator
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Validation data should not be re-generated
# Rescale to 1./255 for all validation image using ImageDataGenerator
test_datagen = image.ImageDataGenerator(rescale=1./255)

# Build Generator by flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir, # Target Directory
    target_size=(150, 150), # Unifying all image sizes to 150*150
    batch_size=20,
    class_mode='binary' # binary label 
)
validation_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)


# Build Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# Compile and fit model
model.compile(
    loss='binary_crossentropy',
    optimizer= optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['acc']
)
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1
)
model.save('ImageGen.h5')
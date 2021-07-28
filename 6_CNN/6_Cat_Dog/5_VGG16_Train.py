import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16

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

# Build Model using VGG16 as convolutional layer base
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False # Do not change weights while training
print(model.summary())

# Compile and fit model
model.compile(
    loss='binary_crossentropy',
    optimizer= optimizers.SGD(learning_rate=0.001, momentum=0.9),
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
model.save('VGG16.h5')
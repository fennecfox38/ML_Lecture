import os
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

# Setting up directory
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(base_dir, 'test')

# test data generator
test_datagen = image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    class_mode='binary',
    shuffle=False
)

# load model
model = models.load_model('CNN.h5')
print(model.summary())

# Evaluate Model using test generator
test_loss, test_acc = model.evaluate(test_generator)
print(test_acc)

# Predict probablities of model using test generator
probabilities = model.predict(test_generator)
print(test_generator.class_indices)

# Print error cases
err = 0
for i, prob in enumerate(probabilities):
        if (prob > 0.5) and ('cat' in test_generator.filenames[i]):
                print(test_generator.filenames[i])
                err += 1
        if (prob < 0.5) and ('dog' in test_generator.filenames[i]):
                err += 1
                print(test_generator.filenames[i])
print(err)
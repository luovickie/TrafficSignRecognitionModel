
import random
import tensorflow as tf
from datasets import load_dataset
import matplotlib.pyplot as plt

def dataset_to_tfdata(dataset, train_split=0.8):
    # Shuffle the dataset manually before processing
    random.shuffle(dataset)
    
    # Split into train and test datasets
    train_size = int(len(dataset) * train_split)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    def generator(dataset):
        for example in dataset:
            image = example['image']
            if not isinstance(image, tf.Tensor):  # Ensure image is a Tensor
                image = tf.keras.preprocessing.image.img_to_array(image)
            label = example['label']
            yield image, label
            
    # Convert to tf.data.Dataset
    train_tfdata = tf.data.Dataset.from_generator(
        lambda: generator(train_dataset),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
    )

    test_tfdata = tf.data.Dataset.from_generator(
        lambda: generator(test_dataset),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
    )
    
    return train_tfdata, test_tfdata

ds = load_dataset("Fuminides/wikiartmini")

dataset = ds['train']
examples = [{'image': example['image'], 'label': example['label']} for example in dataset]

train_ds, test_ds = dataset_to_tfdata(examples)

# Function to preprocess images
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224]) # Resize images to the desired size
    image = image / 255.0  # Normalize the images to [0, 1] range
    return image, label

# Apply preprocessing to the datasets
train_ds = train_ds.map(preprocess).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(27, activation='softmax')  # Adjust the output layer based on the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, epochs=1, validation_data=test_ds)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test accuracy: {test_accuracy:.2f}')

# Function to display a few test images with model predictions
def display_predictions(model, dataset, class_names, num_images=5):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(num_images):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predicted_label = class_names[tf.argmax(predictions[i])]
            true_label = class_names[labels[i]]
            plt.title(f"True: {true_label}\nPred: {predicted_label}")
            plt.axis("off")
    plt.show()

# Assuming you have a list of class names
class_names = ds['train'].features['label'].names

# Display predictions
display_predictions(model, test_ds, class_names)

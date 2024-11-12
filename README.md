# CNN Project - Freddy

Welcome to the **CNN Project - Freddy**! This repository contains a Convolutional Neural Network (CNN) project built to classify images using deep learning techniques. The project is implemented in Python and Jupyter Notebook, leveraging Keras and TensorFlow libraries to build, train, and evaluate the CNN model.

## Project Overview

This project aims to develop a CNN model for image classification. By utilizing convolutional layers, pooling, and fully connected layers, this model learns patterns and features from images, allowing it to accurately classify them into different categories.

### Key Features
- **Data Preprocessing**: Scales images and splits the dataset into training, validation, and testing sets.
- **Model Architecture**: Builds a CNN with convolutional layers, max-pooling, dropout layers for regularization, and dense layers for classification.
- **Training and Evaluation**: Trains the model with image data, monitors performance with metrics, and evaluates accuracy and loss on test data.

## Project Structure

- `Project CNN Jupyter Freddy.ipynb`: The main Jupyter notebook containing code, explanations, and results for each step of the CNN pipeline.
- `data/`: Directory where the image dataset is stored and used for training and testing.
- `models/`: Saved versions of the trained CNN model for reuse.
- `utils/`: Helper functions for data processing and model evaluation.

## Getting Started

### Prerequisites

To run this project, you need:
- Python 3.7 or above
- Jupyter Notebook
- The following Python libraries:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `matplotlib`
  - `sklearn`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/fredsmeds/CNN-Project-Freddy.git
   cd CNN-Project-Freddy
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook "Project CNN Jupyter Freddy.ipynb"
   ```

## Usage

1. **Data Loading**: Load your image dataset into the notebook and preprocess it.
2. **Model Building**: Define the CNN architecture with convolutional, pooling, and dense layers.
3. **Training**: Train the CNN on the preprocessed dataset, monitoring loss and accuracy.
4. **Evaluation**: Use the trained model to evaluate performance on the test dataset.
5. **Prediction**: Test the model on new images to see predictions.

## Example

Below is an example of loading an image and making a prediction with the trained model:

```python
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("models/saved_cnn_model.h5")

# Load and preprocess an image
img_path = "data/sample_image.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make a prediction
prediction = model.predict(img_array)
print("Predicted class:", np.argmax(prediction, axis=1))
```

## Contributing

If you'd like to contribute, please fork the repository and create a pull request. Feel free to open issues for any feature requests, bugs, or improvements.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or questions, please contact **[your email]**.

 # Sign Language to Text and Speech Conversion

Project Overview
This project is designed to bridge the communication gap by recognizing American Sign Language (ASL) gestures and translating them into text and speech. Using Convolutional Neural Networks (CNN), the model can accurately identify hand gestures representing the alphabet (A-Z) and numbers (0-9). Once the gesture is recognized, it is converted to text and then to speech using Google Text-to-Speech (gTTS), making it accessible for verbal communication.

Key Features
ASL Gesture Recognition: The core of the system uses a CNN model to detect ASL gestures, covering both letters and numbers.  
Text-to-Speech Conversion: Once the gesture is recognized, it’s translated into spoken language using gTTS.  
Real-Time Prediction: The system displays predictions instantly, showing both the image and the recognized gesture.  
Efficient Image Preprocessing: Images are resized and normalized for compatibility with the model, ensuring accurate predictions every time.  

Requirements
Before you get started, make sure you have:  
Python 3.x.  
TensorFlow.  
OpenCV.   
gTTS (Google Text-to-Speech).   
IPython.   
numpy.   
pillow.  

Installation Guide
Follow these simple steps to get up and running:

Clone the repository to your local machine:
https://github.com/Poonam-2704/Sign2Speech-Recognition.  
cd sign-language-recognition


Install the required dependencies using pip:
pip install -r requirements.txt


Model Architecture
The model is built to recognize hand gestures corresponding to the ASL alphabet and numbers. Here's a high-level look at the architecture:

Convolutional Layers: These layers extract important features from input images.   
MaxPooling Layers: Reduce the dimensionality of images to enhance performance.   
Dense Layers: Process the extracted features to identify the correct gesture.   
Dropout Layers: Prevent overfitting during training.  
Softmax Output Layer: Classifies gestures into 36 categories (A-Z, 0-9).   
The model is trained using a labeled dataset of ASL images, and once trained, it is saved as asl_model.h5 for easy deployment.   

Training the Model

To train the model, we use labeled ASL images (letters and numbers). The training process utilizes the Adam optimizer and Sparse Categorical Cross-Entropy loss function to ensure accurate predictions. After training, the model is saved for future use and easy inference.

Usage

Once you’ve set up the system, here’s how you can use it to recognize ASL gestures and convert them into speech:

Load the Pre-Trained Model: Load the asl_model.h5 to start recognizing gestures.   
Make Predictions: Feed in images of ASL gestures, and the model will predict the corresponding letter or number.   
Convert to Speech: The recognized gesture is converted into speech using gTTS, making the system interactive and accessible.   
You’ll find more detailed instructions for running the system in the repository.   

Performance Snapshots
Training Graphs
Below is a snapshot of the model’s performance during training, showing the accuracy and loss curves over the epochs.

Accuracy vs Epochs: The graph shows how the model’s accuracy improves with each epoch during training.
Loss vs Epochs: This graph depicts how the model’s loss decreases, indicating improved performance.

![Screenshot 2025-02-03 210421](https://github.com/user-attachments/assets/f83897b1-00bc-4f2d-84a4-651e1ff4a5ad)

Epochs Overview
The model was trained for 10 epochs to reach optimal performance. The number of epochs was determined after careful validation and testing to ensure that the model generalized well without overfitting.

![Screenshot 2025-02-03 210436](https://github.com/user-attachments/assets/1506c0c2-b0dd-48d5-8b70-07daca0665a7)

![Screenshot 2025-02-03 210507](https://github.com/user-attachments/assets/24ca00a7-e202-4e5a-9ae4-08cbaebcc863)



Voice Output
Once the gesture is recognized, it is converted to speech. Below is an example of the speech output for the letter "A":

![Screenshot 2025-02-03 210534](https://github.com/user-attachments/assets/b90d4519-1095-44d3-8410-ed3e95d16726)

Acknowledgements
A big thanks to the tools and libraries that made this project possible:

TensorFlow: For powering the deep learning model that drives gesture recognition.  
OpenCV: For efficient image processing.   
gTTS: For converting the predicted text into speech.   
The ASL Dataset: For providing the necessary data to train the model.  


Contributing

We’d love to see your contributions! Whether it's fixing a bug, adding a new feature, or improving documentation, feel free to fork the repo, create a new branch, and submit a pull request. Please ensure your contributions are well-tested and adhere to the project's coding standards.


Contact

Have questions or feedback? You can open an issue on GitHub, or feel free to reach out to me directly via my GitHub profile.

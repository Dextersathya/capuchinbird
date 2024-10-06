This project aims to classify Capuchinbird calls from audio recordings using a Convolutional Neural Network (CNN) built on TensorFlow and the EfficientNetV2 model. The goal is to identify Capuchinbird calls in long audio recordings, generating spectrograms and using a pre-trained model for transfer learning to classify the bird calls.

Project Structure
capuchinbird.ipynb: Main Jupyter notebook file containing the code for data preparation, model building, training, and testing.
results.csv: Output file containing the predictions for each audio recording in the test set.
/content/Parsed_Capuchinbird_Clips: Directory containing positive audio clips (Capuchinbird calls).
/content/Parsed_Not_Capuchinbird_Clips: Directory containing negative audio clips (not Capuchinbird calls).
/content/Forest Recordings: Directory containing longer audio recordings used for prediction.
Requirements
This project requires the following libraries:

TensorFlow
TensorFlow I/O (tensorflow-io)
Librosa
Matplotlib
NumPy
Kaggle (for downloading the dataset)
To install the dependencies, run:

bash
Copy code
pip install tensorflow tensorflow_io librosa matplotlib numpy kaggle
Dataset
The dataset for this project is obtained from Kaggle. The audio files are classified into:

Positive: Contains Capuchinbird calls.
Negative: Contains other sounds not classified as Capuchinbird calls.
Steps to Run the Code
Download and Unzip the Dataset: The code snippet below downloads the dataset and extracts it.

python
Copy code
!kaggle datasets download -d kenjee/z-by-hp-unlocked-challenge-3-signal-processing
!unzip /content/z-by-hp-unlocked-challenge-3-signal-processing.zip
Load and Process Audio Data: The positive and negative audio files are loaded and converted into spectrograms using Short-Time Fourier Transform (STFT). This ensures compatibility with the CNN model.

Build and Train the Model: The model is built using TensorFlow’s EfficientNetV2 for feature extraction. It is fine-tuned with the spectrogram data, which is converted to 3-channel format to match the expected input shape of the model.

Evaluate the Model: The model is evaluated using Precision, Recall, and Loss metrics. The results are visualized using Matplotlib.

Prediction on Long Audio Files: Long audio recordings are split into 3-second segments, and predictions are made for each segment. The predictions are post-processed to count the number of Capuchinbird calls.

Export Results: The final results are saved into a CSV file named results.csv, listing the number of Capuchinbird calls in each recording.

Usage
Run the notebook step-by-step in a Colab environment to execute the code. Ensure the necessary directories (Parsed_Capuchinbird_Clips, Parsed_Not_Capuchinbird_Clips, and Forest Recordings) exist in your environment.

Post-Processing
The results are post-processed to count consecutive Capuchinbird calls in the long audio files using the following methods:

Groupby Method: Groups consecutive identical predictions to eliminate redundant call counts.
Sum of Calls: Uses the sum of predictions to provide the final count of Capuchinbird calls per recording.
Output
The final results are saved in a CSV file with the following format:

Recording	Capuchin Calls
recording_00.mp3	3
recording_01.mp3	5
...	...
Future Work
Fine-tuning the model further using more data and augmentation techniques.
Implementing more sophisticated methods to identify and isolate bird calls from background noise.
Extending the model to classify multiple species of birds.
License
This project is for educational purposes and is shared under an open-source license. Please check the original Kaggle dataset for any licensing constraints.

Acknowledgements
Kaggle for the dataset.
TensorFlow and EfficientNetV2 for the model architecture.

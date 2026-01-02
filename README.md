AI-Based Parkinson’s Disease Prediction Using Speech Signals
📌 Project Overview

This project presents an AI-based system for Parkinson’s Disease prediction using speech signals. The system analyzes voice recordings and classifies them as Healthy or Parkinson’s Disease using deep learning techniques. The approach is non-invasive, cost-effective, and suitable for early disease screening.

📁 Project Directory Structure
Parkinson_Project/
│
├── dataset/
│   ├── healthy/
│   └── parkinson/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── dataset.py
│   └── config.py
│
├── dataset_statistics.py
├── requirements.txt
├── README.md
└── venv/   (optional – not included in repository)

⚙️ System Requirements

Python 3.8 or higher

Operating System: Windows / Linux / macOS

Minimum 8 GB RAM (16 GB recommended)

GPU recommended for faster training (optional)

🔽 Step 1: Download the Project
Option 1: Download ZIP

Click Code → Download ZIP

Extract the ZIP file to your local system

Option 2: Clone Using Git
git clone https://github.com/your-username/parkinson-disease-prediction.git
cd parkinson-disease-prediction

🐍 Step 2: Create a Python Virtual Environment

Open terminal / command prompt inside the project folder.

python -m venv venv

▶️ Step 3: Activate the Virtual Environment
Windows (Command Prompt)
venv\Scripts\activate

Windows (PowerShell)
venv\Scripts\Activate.ps1

Linux / macOS
source venv/bin/activate


After activation, you should see:

(venv)

📦 Step 4: Install Required Dependencies
pip install -r requirements.txt


If requirements.txt is not available, install manually:

pip install numpy pandas librosa torch torchvision torchaudio scikit-learn matplotlib transformers soundfile

📊 Step 5: Prepare the Dataset

Ensure the dataset is arranged in the following structure:

dataset/
├── healthy/
│   ├── h1.wav
│   ├── h2.wav
│   └── ...
└── parkinson/
    ├── p1.wav
    ├── p2.wav
    └── ...


All audio files must be in .wav format

Folder names must be exactly healthy and parkinson

📈 Step 6: Check Dataset Statistics (Optional)

To view the number of healthy and Parkinson’s voice samples:

python dataset_statistics.py


This will display:

Number of healthy samples

Number of Parkinson’s samples

Total dataset size

🧠 Step 7: Train the Model

Navigate to the source directory and start training:

cd src
python train.py


Training progress (loss and accuracy) will be displayed in the console.

📊 Step 8: Evaluate the Model

After training, evaluate performance metrics:

python evaluate.py


This will generate:

Accuracy

Precision

Recall

F1-score

Confusion matrix

🧪 Step 9: Run Inference (If Implemented)

To test the model on a new audio file:

python predict.py --audio path_to_audio.wav

🛑 Step 10: Deactivate the Virtual Environment

After completing execution:

deactivate

📄 Notes

Ensure the virtual environment is activated before running any scripts

Do not upload the venv folder to GitHub

Keep requirements.txt for reproducibility

GPU is optional but recommended for faster training

🎓 Academic Disclaimer

This project is developed for academic and research purposes only.
It is intended as a supportive screening tool and not a replacement for professional medical diagnosis.

📬 Contact
banerjeeudity@gmail.com

Author: Udity
Degree: B.Tech – Computer Sc

# 🍓 StrawberryTranslate
An automated pipeline for translating manga to English. Uses a YOLOv8 model for bubble detection, manga-ocr for text extraction, and the Helsinki NLP model for high quality translation.

<!-- Replace with a link to your own example image -->

## Features

* **Batch Processing:** Translates all images (`.png`, `.jpg`, `.webp`, `.gif`) in an input folder.
* **Speech Bubble Detection:** Uses a custom-trained YOLOv8 model to find text bubbles.
* **Japanese OCR:** Employs the `manga-ocr` model, specifically trained for reading manga text.
* **High-Quality Translation:** Leverages the `Helsinki-NLP/opus-mt-ja-en` model for accurate and fluent Japanese-to-English translation.
* **Offline First:** After a one-time setup, all models run completely locally without an internet connection.
* **GPU Acceleration:** Automatically uses a CUDA-enabled GPU for both OCR and translation if available, falling back to CPU otherwise.
* **Intelligent Text Placement:** Automatically wraps and resizes translated text to fit neatly inside the original speech bubbles.
* **Dynamic Text Color:** Chooses black or white text for best readability against the bubble's background.

# Setup & Installation
This project only contains the weights required for text detection. All other models, libraries and dependencies will be downloaded in the later steps.

## 0. Prerequisites
* Python 3.10+

* An NVIDIA GPU with CUDA installed (optional, for GPU acceleration)

## 1. Clone the Repository
This will clone the whole repository into a folder named "StrawberryTranslate"
```
git clone https://github.com/sivateja09/StrawberryTranslate.git
```
Move into the folder
```
cd StrawberryTranslate
```
## 2. Set Up a Virtual Environment
A Virtual environment is used to setup and store all libraries and dependencies within the main folder without installing them system wide
It's highly recommended to use a virtual environment.

### 2.1 Create the environment
Inside the main folder open a terminal and run
```   
python -m venv .venv
```
This will create a folder ".venv" which will store all the libraries and dependencies that will be downloaded next

### 2.2 Activate the environment
On Windows:
```
.venv\Scripts\activate
```
On macOS/Linux:
```
source .venv/bin/activate
```
The terminal will now show the env name 

## 3. Install Dependencies
Install all the required Python packages from the requirements.txt file.
```
pip install -r requirements.txt
```

Note: If GPU acceleration isn't working, you may need to install the CUDA version of PyTorch manually. See the PyTorch website for the correct command.

## 4. Download Offline Models
The script uses two external models that need to be downloaded once.

Downloading the OCR Model:
```
python -m manga_ocr
```
(This will download the model and save it to a local cache. It may take a few minutes.)

Downloading the Translation Model:
The transformers library will handle this automatically the first time you run the main script.

## 5. Add Required Files
__The Repository already has the required files for text detection. You can choose your own if you want to__
* YOLO Model: Place your trained speech bubble detection model in the root folder and name it best.pt.
* Font File: Place a TrueType Font file (.ttf) in the root folder and name it font.ttf. This will be used to write the translated text.

## 6. First Run
* Add Images: Place all the manga pages you want to translate into the inputs folder.
* Run the Script: Execute the main Python script from your terminal.
```
python main.py
```
Check the Output: The translated images will be saved in the *output* folder.
The script will process all images from the input folder and save them with a __translated_ suffix.

## Configuration
You can change the behavior of the script by modifying the variables at the top of main.py:
```
YOLO_MODEL_PATH: Path to your custom YOLO model.

INPUT_DIR: Folder to read images from.

OUTPUT_DIR: Folder to save translated images to.

FONT_PATH: Path to the .ttf fon`t file.

TRANSLATION_MODEL_ID: The Hugging Face model to use for translation.
```

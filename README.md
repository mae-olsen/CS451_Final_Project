# CS451_Final_Project
Final Project for UA Data Science Course

The instructions for using this repository and reproducing my results is below. Please keep in mind that you will need to edit hard-coded file and directory path names depending on your local structure. Additionally, parts I-II can be skipped if the intention is to simply use the inference script, as these parts explain how to reproduce the whole pipeline. Skip directly to Part III for inference testing.


Part I: Data Cleaning, Exploration, and Augmentation

The individual symbol images are contained in symbols/extracted_images directory.

1. Use gen_expressions.py script to generate 10,000 synthetic whiteboard images of randomly placed symbols
This will create synthetic images with corresponding label files that contain all symbol bounding box coordinates with the symbol class.
3. Run change_labels.py script to mask symbol classes (YOLOv8n model is only to detect symbol or no symbol rather than the symbol type)
4. Use data_explore.ipynb for exploratory data analysis and 80/20 train/val data splitting


Part II: Model Training

1. To fine-tune the YOLOv8n model, create .yaml file, like yolo.yaml with your directory structure and run yolo_tuning.ipynb. This notebook also produces some visualizations based on results. The best.pt model weights file will be saved after fine-tuning the model.
2. For HPO and training for the CNN and logistic regression models, use class_models.ipynb. This file references int_mapping.json, which maps integers to LaTeX symbol labels.
 a. The logistic regression model weights will be saved as lr_model.pkl
 b. The CNN model weights will be saved as cnn_model_weights.pth
3. For model validation, use full_pipeline.py script


**Part III: Inference**

The requirements.txt file shows libraries you will need to run inference. Also note that to use pdflatex in inference.py, a local LaTeX distribution must be installed and ensure pdflatex is on PATH.

1. Make sure to have best.pt, lr_model.pkl, and cnn_model_weights.pth in your directory
2. Create a subdirectory of images you would like to use for inference. The script will iterate through each image and output that as a new line in the LaTeX file. Note: only one expression line per image. The models can only handle that format currently. expression.jpg is an example of an image that can be used for inference.
3. Run **python3 inference.py --img_dir <img_dir_name>**. This will output results in terminal as well as output.tex and output.pdf.

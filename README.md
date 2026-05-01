# CS451_Final_Project
Final Project for UA Data Science Course

The instructions for using this repository and reproducing my results is below. Please keep in mind that you will need to edit hard-coded file and directory path names depending on your local structure.

Part I: Data Cleaning, Exploration, and Augmentation
The individual symbol images are contained in symbols/extracted_images directory.

1. Use gen_expressions.py script to generate 10,000 synthetic whiteboard images of randomly placed symbols
This will create synthetic images with corresponding label files that contain all symbol bounding box coordinates with the symbol class.
3. Run change_labels.py script to mask symbol classes (YOLOv8n model is only to detect symbol or no symbol rather than the symbol type)
4. Use data_explore.ipynb for exploratory data analysis and 80/20 train/val data splitting

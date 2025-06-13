# Chest X-ray Pneumonia Detection with Deep Learning

This project implements an end-to-end deep learning pipeline for automatic detection of pneumonia from chest X-ray images. Using transfer learning with PyTorch and a pre-trained ResNet18 model, the system classifies chest X-rays as NORMAL or PNEUMONIA. The project also features a user-friendly Streamlit web app for fast, local inference.

## Features

- **Data Preparation**: Guide and notebook (Colab) for dataset preprocessing and augmentation.
- **Model Training**: PyTorch-based transfer learning with ResNet18.
- **Evaluation**: Classification report, confusion matrix, ROC curve, and Precision-Recall curve.
- **Deployment**: Streamlit app for local image upload and instant prediction.
- **Clean Structure**: Modular folders for models, notebooks, and app code.

## Quick Start

1. Clone the Repository

```
git clone git@github.com:mohanseetha/pneumonia-prediction.git
cd pneumonia-prediction
```

2. Install Dependencies

```
pip install torch torchvision streamlit scikit-learn matplotlib pillow
```

3. Open `⁠notebooks/chest_xray_training.ipynb⁠` in Google Colab or Jupyter.

4. Follow the instructions to prepare the data, train, and evaluate the model.

5. Download the resulting `⁠.pth⁠` file and place it in the `⁠models/`⁠ folder.

6. Run the Streamlit App

```
streamlit run app.py
```

Open http://localhost:8501 in your browser. Upload a chest X-ray image and receive a prediction: NORMAL or PNEUMONIA.

## References

- [Chest X-ray Images (Pneumonia) Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Note

This project is for educational and research purposes only.
Please consult a medical professional for any health-related decisions.

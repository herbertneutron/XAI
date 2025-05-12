# Brain Tumor Classification with CNN and Grad-CAM

This repository contains a deep learning model for classifying brain tumor MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. The project includes model training, evaluation, and Explainable AI (XAI) techniques using Grad-CAM to visualize model attention.

## Features

- Custom CNN architecture for brain tumor classification
- Data loading and preprocessing pipeline
- Model training and validation
- Performance evaluation with metrics (accuracy, confusion matrix)
- Grad-CAM implementation for model interpretability
- Model saving and loading functionality

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy
- OpenCV (cv2)
- seaborn (for visualization)

Install requirements with:
```bash
pip install torch torchvision matplotlib scikit-learn numpy opencv-python seaborn
```

## Dataset

The model uses the Brain Tumor MRI Dataset, which contains four classes:
1. Glioma
2. Meningioma
3. No tumor
4. Pituitary tumor

The dataset should be organized in the following structure:
```
/content/
  ├── Training/
  │   ├── glioma/
  │   ├── meningioma/
  │   ├── no_tumor/
  │   └── pituitary/
  └── Testing/
      ├── glioma/
      ├── meningioma/
      ├── no_tumor/
      └── pituitary/
```

## Usage

1. **Data Preparation**:
   - The code automatically unzips and loads the dataset from Google Drive (modify paths as needed)
   - Performs train/validation split (80/20)

2. **Model Training**:
   ```python
   model = BrainTumorCNN().cuda()
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
   ```

3. **Evaluation**:
   - Test accuracy:
     ```python
     evaluate_model(model, test_loader)
     ```
   - Detailed metrics:
     ```python
     evaluate_model_with_metrics(model, test_loader, class_names)
     ```

4. **Grad-CAM Visualization**:
   ```python
   # Load and preprocess an image
   image = Image.open(img_path).convert("RGB")
   input_tensor = transform(image).unsqueeze(0).cuda()
   
   # Generate and display Grad-CAM
   gradcam = generate_gradcam(model, input_tensor)
   display_gradcam(input_tensor.squeeze(0), gradcam)
   ```

5. **Model Saving/Loading**:
   - Save:
     ```python
     torch.save(model.state_dict(), 'brain_tumor_cnn.pth')
     ```
   - Load:
     ```python
     model = BrainTumorCNN().cuda()
     model.load_state_dict(torch.load('brain_tumor_cnn.pth'))
     ```

## Results

The model provides:
- Classification predictions
- Accuracy metrics
- Confusion matrix visualization
- Grad-CAM heatmaps showing which image regions influenced the model's decisions

## File Structure

- `XAI.ipynb`: Main Jupyter notebook containing all code
- `/content/Training/`: Training dataset
- `/content/Testing/`: Testing dataset
- `brain_tumor_cnn.pth`: Saved model weights

## Note

This implementation is optimized for Google Colab with GPU acceleration. For local execution, modify the paths and ensure CUDA is available if using GPU.

## License

This project is open-source and available for educational and research purposes.

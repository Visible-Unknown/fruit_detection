# 🍎🥭 Fruit Classification using CNN

A Convolutional Neural Network (**CNN**) built with **TensorFlow/Keras** to classify **36 types of fruits and vegetables**.  
The dataset was stored in **Google Drive** and training was performed in **Google Colab**.  

---

## 📂 Dataset
- **Training set**: 3115 images (36 classes)  
- **Validation set**: 351 images (36 classes)  
- **Test set**: 359 images (36 classes)  

Each class (e.g., `apple`, `banana`, `mango`) has its own folder.  

---

## 🔧 Project Workflow
1. **Mount Google Drive** → Access dataset stored in `Fruit/train`, `Fruit/validation`, `Fruit/test`.  
2. **Data Preprocessing** → Images resized to `(64x64)` and loaded into batches.  
3. **Model Building**  
   - Conv2D + MaxPooling layer  
   - Dropout (0.5) to prevent overfitting  
   - Flatten → Dense layer with 36 output units (`softmax`)  
4. **Compilation** → Optimizer: `Adam`, Loss: `Categorical Crossentropy`, Metric: `Accuracy`.  
5. **Training** → Model trained for **32 epochs**.  
6. **Evaluation** → Performance tested on validation & test sets.  
7. **Saving Model** → Saved as `trained_model.h5`.  
8. **Prediction** → Model tested on a single image (example: correctly predicted "apple").  

---

## 📊 Results
- **Validation Accuracy**: ~92.3%  
- **Test Accuracy**: ~92.2%  

Training & validation accuracy showed consistent improvement across epochs.  

---

## 📈 Visualization
- **Training Accuracy Curve** → Shows accuracy improvement over epochs.  
- **Validation Accuracy Curve** → Demonstrates strong generalization to unseen data.  

---

## 💾 Outputs
- `trained_model.h5` → Saved CNN model.  
- `training_history.json` → Accuracy & loss values for each epoch.  

---

## ⚡ Requirements
- Python 3.x  
- TensorFlow 2.x  
- Matplotlib  
- OpenCV  
- Google Colab / Jupyter Notebook  

---

## 🎯 Future Enhancements
- Add **data augmentation** for more robust training.  
- Use **transfer learning models** (e.g., VGG16, ResNet50).  
- Deploy as a **web application** for real-time fruit classification.  

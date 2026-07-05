Smart Kitchen Monitoring System using CNN:

Overview:
This project implements a Convolutional Neural Network (CNN) to classify food into three freshness categories: Fresh, Warning (Near Expiry), and Spoiled. The project uses a synthetically generated dataset along with simulated IoT sensor data to demonstrate how AI can support food quality monitoring and reduce food waste in restaurant kitchens.

Features:
-Generated a synthetic dataset of 10,000 food samples
-Simulated IoT sensor data:
-Temperature
-Humidity
-CO₂ concentration
-Storage location
-Hours since delivery
-Remaining shelf life
-Visual mold score
-Trained a CNN model using PyTorch

Classified food into:
-Fresh
-Warning
-Spoiled

Also , I have saved the trained model for future predictions

Technologies Used:
-Python
-PyTorch
-NumPy
-Pandas
-Pillow (PIL)
-Scikit-learn

Project Workflow:

Synthetic Food Images
        ↓
IoT Sensor Data Simulation
        ↓
Dataset Generation
        ↓
CNN Model Training
        ↓
Food Freshness Classification
        ↓
Fresh | Warning | Spoiled

Future Improvements':
1 Use real food image datasets
2 Integrate real IoT sensors
3 Develop a Digital Twin dashboard
4 Deploy as a web application

Author: Shashank Upadhayay

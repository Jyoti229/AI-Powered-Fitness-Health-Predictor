# AI_Powered Fitness & Health Predictor

An AI-powered web application that predicts an individual’s fitness level and health condition using lifestyle and physiological data. The project demonstrates end-to-end machine learning pipeline development — from data generation to web integration using Flask.

---

## 1. Problem Statement

In the modern world, personal health monitoring has become essential. Many individuals lack awareness about their health or fitness status due to unavailability of predictive systems. This project solves that gap by providing real-time predictions of:

- Fitness Level (Fit, Average, Unfit)
- Health Condition (Healthy, Moderate, Poor)

using user input parameters like age, gender, weight, height, lifestyle habits, and occupation.

---

## 2. Key Features

- Predicts both fitness and health status using ML
- Multi-output Random Forest regression model
- Custom data simulation of 150+ records
- Role-based lifestyle inference: student, employee, private owner
- Flask-powered web UI with form input
- Modular code structure and scalable architecture

---

## 3. Technologies Used

| Layer         | Tools / Libraries                      |
|---------------|-----------------------------------------|
| Language      | Python 3                                |
| Libraries     | scikit-learn, pandas, numpy             |
| Model         | MultiOutputRegressor with RandomForest  |
| Web App       | Flask, HTML (Jinja2)                    |
| Data Handling | LabelEncoder, StandardScaler            |
| Development   | VS Code                                 |

---

## 4. Project Structure

fitness_health_predictor/
│
├── app.py # Flask application script
├── templates/
│ └── index.html # Web form
├── model/
│ ├── multioutput_rf_model.pkl
│ ├── scaler.pkl
│ ├── le_gender.pkl
│ ├── le_activity.pkl
│ ├── le_diet.pkl
│ ├── le_role.pkl
│ ├── le_fitness.pkl
│ └── le_health.pkl
├── requirements.txt # List of Python dependencies

## 5. Dataset Overview

The dataset used for this project consists of **synthetically generated data** for over **150 individuals**. It includes features related to **demographics, lifestyle, and health**, designed to simulate real-world behavior and support accurate predictions.

### Features

| Feature Name        | Description                                         | Type          | Example                     |
|---------------------|-----------------------------------------------------|---------------|-----------------------------|
| `Age`               | Age of the individual                               | Numerical     | 22, 35, 45                  |
| `Gender`            | Biological gender                                   | Categorical   | Male, Female                |
| `Height`            | Height in centimeters                               | Numerical     | 165, 178                    |
| `Weight`            | Weight in kilograms                                 | Numerical     | 58, 70                      |
| `Role`              | Type of work/lifestyle role                         | Categorical   | Student, Employee, Private Owner |
| `Diet`              | Type of regular diet                                | Categorical   | Balanced, Vegetarian, Fast Food |
| `Physical Activity` | Average level of physical activity                  | Categorical   | High, Medium, Low           |

### Target Labels

| Label Name       | Description                                      | Values                     |
|------------------|--------------------------------------------------|----------------------------|
| `Fitness Level`  | Indicates overall fitness of the individual      | Fit, Average, Unfit        |
| `Health Status`  | Indicates general health condition               | Healthy, Moderate, Poor    |

---

### Sample Records

| Age | Gender | Height | Weight | Role         | Diet      | Activity | Fitness | Health   |
|-----|--------|--------|--------|--------------|-----------|----------|---------|----------|
| 21  | Female | 160    | 50     | Student      | Balanced  | Medium   | Fit     | Healthy  |
| 35  | Male   | 175    | 78     | Employee     | Fast Food | Low      | Average | Moderate |
| 42  | Female | 168    | 85     | Private Owner| Vegetarian| Low      | Unfit   | Poor     |

---

### 6. Data Preprocessing Steps

- **Label Encoding** for categorical features (`Gender`, `Role`, `Diet`, `Activity`)
- **Standard Scaling** for numerical features (`Age`, `Height`, `Weight`)
- **MultiOutput labels** (`Fitness`, `Health`) encoded and predicted
- Data split into training and testing sets for evaluation

---

## 7. Results and Outcomes

The model predicts Fitness Level and Health Status with good accuracy:

- Fitness Level Accuracy: 85%  
- Health Status Accuracy: 82%

Precision, recall, and F1-scores are all around 0.8, showing balanced performance.

This shows the model effectively uses input features to estimate health and fitness levels.


# Vehicle Insurance Response Predictor

A machine learning application that predicts whether a customer will purchase vehicle insurance based on various demographic and vehicle-related features.

##  Features

- **Machine Learning Model**: Trained model to predict insurance purchase likelihood
- **Flask API**: RESTful API for predictions
- **Streamlit Web App**: Interactive web interface for easy predictions
- **Data Analysis**: Comprehensive analysis of insurance response patterns

## Dataset

The project uses a dataset containing 381,109 records with the following features:
- Gender
- Age
- Driving License status
- Region Code
- Previously Insured status
- Vehicle Age
- Vehicle Damage history
- Annual Premium
- Policy Sales Channel
- Customer Vintage

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Caxive/repo-ml
   cd repo-ml
   python -m venv venv
   # Activate virtual environment
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### Method 1: Using the Streamlit Web App (Recommended)

1. **Start the Flask API server**
   ```bash
   python app.py
   ```

2. **In a new terminal, start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

### Method 2: Using the Flask API directly

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Make a POST request** to `http://127.0.0.1:5000/predict` with the following JSON structure:
   ```json
   {
     "gender": "Male",
     "age": 25,
     "dl": "Yes",
     "reg_code": 28,
     "prev_insured": "No",
     "vehicle_age": "1-2 Years",
     "vehicle_damage": "Yes",
     "annual_premium": 50000,
     "sales_channel": 152,
     "vintage": 217
   }
   ```

##  Project Structure

```
Insurnce-Response-Predictor/
├── app.py                                    # Flask API server
├── streamlit_app.py                         # Streamlit web application
├── Vehicle Insurance Response Predictor using Machine Learning.ipynb  # Jupyter notebook with analysis
├── insurance-response-predictor.pkl         # Trained ML model
├── requirements.txt                         # Python dependencies
├── README.md                               # Project documentation
├── dataset/
│   └── train.csv                           # Training dataset
└── LICENSE                                 # MIT License
```

## Model Details

The model uses machine learning algorithms to predict insurance purchase responses based on customer demographics and vehicle information. The model is trained on historical data and provides predictions with the following features:

- **Input Features**: 11 features including gender, age, vehicle details, and customer history
- **Output**: Binary prediction (Will buy insurance / Will not buy insurance)

##  API Endpoints

### POST /predict

Predicts insurance purchase likelihood for a given customer.

**Request Body:**
- `gender` (string): "Male" or "Female"
- `age` (integer): Customer age
- `dl` (string): "Yes" or "No"
- `reg_code` (integer): Region code
- `prev_insured` (string): "Yes" or "No"
- `vehicle_age` (string): "Less than 1 Year", "1-2 Years", or "More than 2 Years"
- `vehicle_damage` (string): "Yes" or "No"
- `annual_premium` (float): Annual premium amount
- `sales_channel` (integer): Policy sales channel code
- `vintage` (integer): Customer vintage

**Response:**
```json
{
  "prediction": "The customer will buy the insurance"
}
```

## Use Cases

- **Insurance Companies**: Predict customer response to insurance offers
- **Marketing Teams**: Identify potential customers for targeted campaigns
- **Sales Teams**: Prioritize leads based on purchase likelihood
- **Analytics Teams**: Understand factors influencing insurance purchases

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Dataset sourced for machine learning training
- Flask and Streamlit communities for excellent documentation
- Scikit-learn team for the ML framework

##  Support

If you have any questions or need help with the project, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the author

---


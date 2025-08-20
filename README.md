# 💳 CodeAlpha Credit Scoring Model

[![CI/CD Pipeline](https://github.com/dkumi12/CodeAlpha_CreditScoreModel/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dkumi12/CodeAlpha_CreditScoreModel/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Professional machine learning system for credit score classification with comprehensive validation and modular architecture**

## 🚀 Features

### **Core ML Functionality**
- **🤖 Random Forest Classification**: Advanced ensemble model for credit score prediction
- **📊 Comprehensive Preprocessing**: Automated categorical encoding and feature scaling
- **🎯 Multi-Class Classification**: Predicts Low, Average, and High credit scores
- **⚖️ Model Evaluation**: Complete performance metrics and validation

### **Professional Engineering**
- **✅ Modular Architecture**: Clean separation of concerns with shared utilities
- **✅ Comprehensive Validation**: Input validation and error handling throughout
- **✅ CLI Interface**: Command-line interface with configurable parameters
- **✅ Automated Testing**: Complete test suite with pytest
- **✅ CI/CD Pipeline**: GitHub Actions with testing and security scanning
- **✅ Code Quality**: Linting, formatting, and type checking

### **Financial Domain Expertise**
- **💰 Credit Risk Assessment**: Industry-standard classification approach
- **📈 Feature Engineering**: Proper categorical variable handling
- **🔍 Hyperparameter Optimization**: GridSearchCV for optimal performance
- **📊 Performance Metrics**: ROC-AUC, accuracy, and classification reports

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dkumi12/CodeAlpha_CreditScoreModel.git
cd CodeAlpha_CreditScoreModel

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python src/credit_scoring_model.py
```

## 📖 Usage

### **Basic Training**
```bash
# Train model with default settings
python src/credit_scoring_model.py

# Train with custom parameters
python src/credit_scoring_model.py --test-size 0.3 --random-state 123

# Skip hyperparameter tuning for faster training
python src/credit_scoring_model.py --no-tuning
```

### **Advanced Configuration**
```bash
# Custom data path and model directory
python src/credit_scoring_model.py \\
    --data-path "Data/Custom_Dataset.csv" \\
    --model-dir "custom_models" \\
    --test-size 0.25
```

### **Programmatic Usage**
```python
from src.utils import (
    load_and_validate_dataset,
    preprocess_credit_data,
    validate_input_data
)

# Load and process data
data = load_and_validate_dataset('Data/Credit Score Classification Dataset.csv')
processed_data, info = preprocess_credit_data(data)

# Validate new input
sample_input = {
    'Age': 35,
    'Income': 75000,
    'Education': 'Graduate',
    'Marital Status': 'Married',
    'Gender': 'Female',
    'Home Ownership': 'Own'
}

is_valid, message = validate_input_data(sample_input)
print(f"Input validation: {is_valid} - {message}")
```
## 🏗️ Project Structure

```
CodeAlpha_CreditScoringModel/
├── src/                           # Source code modules
│   ├── credit_scoring_model.py   # Enhanced main training script
│   └── utils.py                  # Shared utilities and validation
├── Src/                          # Original source (maintained for compatibility)
│   └── credit_scoring_model.py  # Original monolithic script
├── Data/                         # Dataset storage
│   └── Credit Score Classification Dataset.csv
├── Models/                       # Trained model artifacts (created during training)
├── tests/                        # Comprehensive test suite
│   ├── __init__.py
│   └── test_utils.py            # Unit tests for utilities
├── .github/workflows/            # CI/CD automation
│   └── ci-cd.yml               # GitHub Actions pipeline
├── logs/                        # Application logging (created during runtime)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 📊 Model Performance

### **Dataset Information**
- **Source**: Credit Score Classification Dataset
- **Features**: Age, Income, Education, Marital Status, Gender, Home Ownership
- **Target Classes**: Low (0), Average (1), High (2) credit scores
- **Preprocessing**: One-hot encoding for categorical variables, StandardScaler normalization

### **Model Specifications**
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, ROC-AUC (multi-class), Classification Report
- **Train/Test Split**: 80/20 (configurable)

### **Expected Performance**
- **Accuracy**: >85% on test set
- **ROC-AUC**: >0.85 (multi-class one-vs-rest)
- **Balanced Performance**: Good precision and recall across all credit score classes

## 🧪 Development

### **Running Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### **Code Quality**
```bash
# Install linting tools
pip install flake8 black isort

# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Sort imports
isort src/ tests/
```

### **Model Training Options**
```bash
# Quick training (no hyperparameter tuning)
python src/credit_scoring_model.py --no-tuning

# Custom test split
python src/credit_scoring_model.py --test-size 0.3

# Different random seed
python src/credit_scoring_model.py --random-state 123
```

## 🔧 Technical Improvements

### **What's New in This Version**

✅ **Modular Architecture**: Broke down monolithic script into reusable utilities  
✅ **Enhanced Error Handling**: Comprehensive exception handling throughout  
✅ **Input Validation**: Robust validation for all data inputs  
✅ **CLI Interface**: Command-line interface with configurable parameters  
✅ **Professional Structure**: Organized codebase following best practices  
✅ **Comprehensive Testing**: Unit tests for all utility functions  
✅ **CI/CD Pipeline**: Automated testing and security scanning  
✅ **Documentation**: Professional README with usage examples  

### **Code Quality Improvements**
- **Eliminated Monolithic Design**: Original single file → modular architecture
- **Added Comprehensive Logging**: Track training progress and debugging
- **Implemented Data Validation**: Ensure data quality and format compliance
- **Enhanced Error Messages**: User-friendly error reporting
- **Standardized File Paths**: Configurable paths instead of hardcoded values

## 🔒 Security & Compliance

### **Data Protection**
- **Input Validation**: All inputs validated for type and range
- **Error Handling**: Graceful handling of malformed data
- **Path Security**: Validated file paths to prevent directory traversal
- **Dependency Security**: Automated vulnerability scanning with Safety and Bandit

### **Model Security**
- **Model Validation**: Automated testing of saved models
- **Reproducibility**: Fixed random seeds for consistent results
- **Audit Trail**: Comprehensive logging for model training process

## 🚀 Deployment

### **Local Development**
```bash
# Run with development logging
python src/credit_scoring_model.py --data-path Data/Credit\\ Score\\ Classification\\ Dataset.csv
```

### **Production Considerations**
- Use environment variables for sensitive configuration
- Implement proper logging in production environment
- Consider model versioning for production deployments
- Set up monitoring for model performance degradation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/`)
5. Ensure code quality (`flake8 src/`, `black src/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure CI/CD pipeline passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 CodeAlpha Internship Project

This project was developed as part of the CodeAlpha Data Science internship program, demonstrating:
- **Professional ML Development**: Industry-standard practices and methodologies
- **Code Quality**: Clean, maintainable, and well-documented code
- **Testing**: Comprehensive validation and quality assurance
- **Documentation**: Professional presentation and usage guides

## 📞 Support

For questions or issues:
- Open an [issue](https://github.com/dkumi12/CodeAlpha_CreditScoreModel/issues)
- Contact: [@dkumi12](https://github.com/dkumi12)

---

**Developed with ❤️ as part of CodeAlpha Data Science Internship**
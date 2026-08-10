# Craylife Flask Project

## Overview
The Craylife Flask project is a web application designed for managing and classifying crayfish data using machine learning. It provides a user-friendly interface for authentication, data entry, and visualization of results.

## Project Structure
```
craylife_flask-1
├── app
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   ├── forms.py
│   ├── utils.py
│   ├── MLtrain
│   │   ├── MaleFemale_Crayfish_Model.h5
│   │   ├── MaleFemale-detector.ipynb
│   │   ├── Crayfish_train
│   │   │   ├── Female
│   │   │   └── Male
│   │   └── Crayfish_validation
│   │       ├── Female
│   │       └── Male
│   ├── static
│   │   ├── images
│   │   ├── css
│   │   │   └── style.css
│   │   └── js
│   │       └── script.js
│   └── templates
│       ├── base.html
│       ├── auth
│       │   ├── login.html
│       │   ├── password_reset_complete.html
│       │   ├── password_reset_confirm.html
│       │   ├── password_reset_done.html
│       │   └── password_reset_form.html
│       └── dashboard
│           ├── activity.html
│           ├── batch_form.html
│           ├── batch_sale.html
│           ├── capture.html
│           ├── classify.html
│           ├── home.html
│           ├── inventory.html
│           ├── model_confirm_delete.html
│           ├── model_list.html
│           ├── reports.html
│           └── settings.html
├── migrations
├── app.py
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

## Setup Instructions
1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd craylife_flask-1
   ```

2. **Create a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Database Migration**
   Initialize the database and apply migrations:
   ```
   flask db init
   flask db migrate
   flask db upgrade
   ```

5. **Run the Application**
   ```
   python run.py
   ```

## Usage
- Navigate to `http://127.0.0.1:5000` in your web browser to access the application.
- Use the authentication pages to log in or reset your password.
- Explore the dashboard for data entry and classification functionalities.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
# 🗣️ Darija to English Translator Web App (Final year project at University)

This is a full-stack web application built with **Django**, **Python**, **HTML**, **CSS**, and **JavaScript** that translates Moroccan Darija (a dialect of Arabic) into English.

Registered users can log in to use the translator, view previous translations, and explore an intuitive, minimal interface.

---

## 📸 Preview

![Screenshot of the Darija to English Translator Web App](screenshot.png)

---

## ✨ Features

- 🔐 **User Registration & Login**  
  Create an account and log in securely using Django's built-in authentication system.

- 🌍 **Darija to English Translationa and vice versa**  
  Switch between Morrocan Arabic to English and English to Moroccan Arabic translations using a trained NLP model.

- 📜 **Translation History**  
  Logged-in users can view their previous translations.

- 🎨 **Clean, Responsive Interface**  
  Styled with HTML, CSS, and JavaScript to provide a clean and interactive user experience.

---

## 🚀 Technologies Used

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Authentication:** Django Auth
- **Database:** SQLite (default) or MySQL (optional)
- **Model:** Pre-trained NLP model (DarijaBERT or similar)

---

## 🛠️ Getting Started

### 🔧 Prerequisites

- Python 3.8+
- `pip` or `venv`
- MySQL
- Git

### 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/darija-translator.git

```
2. pip install mysqlclient
3. pip install torch
4. pip install transformers
5. pip install nltk
6. pip install pydoda
7. download the mysql image on docker and run as a container on port 3308
8. Add a new connection on mysql workbench on local host port 3308
9. cd darija
10. python manage.py runserver

This project is a chatbot built using the **LlamaIndex** and **Gemini** frameworks, with the Indian Constitution preloaded as the data source.

## Setup Instructions

### 1. Create a Virtual Environment (venv)
To create a virtual environment, run the following command:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows**: 
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**: 
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Add Your Gemini API Key
Create a `.env` file in the project root directory and add your **Gemini API key** in the following format:
```
GEMINI_API_KEY=your-api-key-here
```

### 5. Running the Application
To run the chatbot application, execute the following command:
```bash
python -m streamlit run chatbot.py
```

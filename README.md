Here’s your updated `README.md` file with your **live demo link** added and ensuring it aligns with the submission guidelines (e.g., code upload instructions, live link, methodologies, sample outputs, etc.).

---

# Medical Transcript Analyzer

A Streamlit web application that analyzes medical conversation transcripts to extract structured information, perform sentiment analysis, and generate SOAP notes for physicians.

[**Live Demo**](https://physiciannotetaker-mbnftxevfg9khqm3eser5q.streamlit.app)

---

## Features

- **Medical Entity Extraction**: Identifies symptoms, treatments, diagnoses, and timeframes.
- **Sentiment Analysis**: Determines patient emotional state and communicative intent.
- **Structured Summarization**: Creates a concise patient summary.
- **SOAP Note Generation**: Produces a standardized clinical note format.
- **Batch Processing**: Analyze multiple transcripts at once.

---

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/shiva-sai-824/physician-notetaker.git
   cd physician-notetaker
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. The application will be available at [http://localhost:8501](http://localhost:8501).

---

## Usage

1. Enter or upload a medical transcript in the input area.
2. Alternatively, click the "Use sample transcript" checkbox to load a demonstration transcript.
3. Click the **Analyze Transcript** button to process the input.
4. View the results in four categories:
   - **Medical Details**: Extracted entities.
   - **Patient Summary**: Structured information.
   - **Sentiment Analysis**: Patient's emotional state.
   - **SOAP Note**: Generated clinical documentation.
5. Download the analysis results as a JSON file if needed.

---

## Batch Processing

For processing multiple transcripts at once:

1. Navigate to the **Batch Processing** tab.
2. Upload multiple text files containing transcripts.
3. Click **Process Batch** to analyze all files.
4. View the summary table and download the complete results as JSON.

---

## Methodology

### 1. Named Entity Recognition (NER)
- Uses SpaCy's pre-trained model to identify medical entities.
- Enhanced with rule-based pattern matching for medical terminology.
- Extracts symptoms, treatments, diagnoses, and temporal information.

### 2. Sentiment Analysis
- Employs the Hugging Face Transformers library with the `DistilBERT` model.
- Classifies patient sentiment as "Anxious," "Neutral," or "Reassured."
- Rule-based intent detection identifies communicative purposes.

### 3. Structured Summarization
- Converts extracted entities and relationships into a structured patient profile.
- Generates a comprehensive yet concise summary of the patient's condition.

### 4. SOAP Note Generation
- Produces standardized clinical documentation following the SOAP format:
  - **S**ubjective: Patient's symptoms and statements.
  - **O**bjective: Observable findings from examination.
  - **A**ssessment: Clinical assessment and diagnosis.
  - **P**lan: Treatment plan and follow-up recommendations.

---

## Sample Output

### Medical Details
```json
{
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Treatment": ["Physiotherapy sessions", "Painkillers"],
  "Diagnosis": ["Whiplash injury"],
  "Timeframes": ["September 1st", "four weeks"]
}
```

### SOAP Note
```json
{
  "Subjective": {
    "Chief_Complaint": "Neck pain, Back pain, Head impact",
    "History_of_Present_Illness": "Patient involved in a car accident in September"
  },
  "Objective": {
    "Physical_Exam": "Physical examination mentioned, details not provided",
    "Observations": "Based on patient's statements"
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury",
    "Severity": "Improving based on patient statements"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy, use painkillers as needed",
    "Follow_Up": "As needed based on symptom progression"
  }
}
```

---

## Screenshots

Here are some examples of the application output:

1. **Medical Details Extraction**:

   ![Medical Details Extraction](https://drive.google.com/uc?id=17tatUOWmdbL7afE_AnK01Ye3hBBGIGyL)

2. **Sentiment Analysis**:

   ![Sentiment Analysis](https://drive.google.com/uc?id=1Pins2Qk79wXS8RsgVc6eLfRMCYoenaI9)

3. **SOAP Note Generation**:

   ![SOAP Note Generation](https://drive.google.com/uc?id=1hWFotwhTt1Fb7_CY3PWKE8xWu11yp6Tl)

---

## Technologies Used

- **SpaCy**: For named entity recognition and linguistic analysis.
- **Transformers (HuggingFace)**: For sentiment analysis.
- **Streamlit**: For the web application interface.
- **Pandas**: For data manipulation and display.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Developed by **Shiva Sai**

GitHub: [shiva-sai-824](https://github.com/shiva-sai-824)  
Live App: [**Physician Notetaker**](https://physiciannotetaker-mbnftxevfg9khqm3eser5q.streamlit.app)

---


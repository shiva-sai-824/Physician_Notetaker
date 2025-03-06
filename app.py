import streamlit as st
import spacy
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
import json
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Medical Transcript Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextInput, .stTextArea {
        padding: 1rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .header-text {
        color: #0066cc;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader-text {
        color: #0066cc;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='header-text'>Medical Transcript Analyzer</div>", unsafe_allow_html=True)
st.markdown("""
This application analyzes medical conversation transcripts to extract structured information, 
perform sentiment analysis, and generate SOAP notes for physicians.
""")

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load NLP models in a function to allow for caching and progress indication
def load_models():
    with st.spinner('Loading NLP models... This may take a minute.'):
        progress_bar = st.progress(0)
        
        # Load SpaCy model
        try:
            progress_bar.progress(10)
            st.session_state.spacy_model = spacy.load("en_core_web_md")
            progress_bar.progress(50)
        except OSError:
            st.warning("SpaCy model not found. Downloading now...")
            os.system("python -m spacy download en_core_web_md")
            st.session_state.spacy_model = spacy.load("en_core_web_md")
            progress_bar.progress(50)
        
        # Load Hugging Face sentiment model
        progress_bar.progress(60)
        st.session_state.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased")
        progress_bar.progress(100)
        
        st.session_state.models_loaded = True
        time.sleep(0.5)  # Brief pause to show completed progress
        progress_bar.empty()
        st.success("Models loaded successfully!")

# Extract medical details - adapted from physician_notetaker.ipynb
def extract_medical_details(transcript_text):
    doc = st.session_state.spacy_model(transcript_text)
    symptoms = []
    treatments = []
    diagnosis = []
    timeframes = []

    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            timeframes.append(ent.text)

    # Manual overrides for transcript based on keyword matching
    # This approach mimics the manual overrides in your notebook
    if "pain" in transcript_text.lower() or "discomfort" in transcript_text.lower():
        symptoms.append("Pain/Discomfort")
        
    if "neck" in transcript_text.lower():
        symptoms.append("Neck pain")
        
    if "back" in transcript_text.lower():
        symptoms.append("Back pain")
        
    if "hit my head" in transcript_text.lower():
        symptoms.append("Head impact")
        
    if "physiotherapy" in transcript_text.lower():
        treatments.append("Physiotherapy sessions")
        
    if "painkiller" in transcript_text.lower():
        treatments.append("Painkillers")
        
    if "whiplash" in transcript_text.lower():
        diagnosis.append("Whiplash injury")

    return {
        "Symptoms": list(set(symptoms)),
        "Treatment": list(set(treatments)),
        "Diagnosis": list(set(diagnosis)),
        "Timeframes": list(set(timeframes))
    }

# Structured summary - adapted from physician_notetaker.ipynb
def structured_summary(medical_details, transcript_text):
    # This function mirrors the structured_summary function from your notebook
    patient_name = "Unknown"
    
    # Extract patient name
    if "Ms. Jones" in transcript_text:
        patient_name = "Ms. Jones"
    
    # Determine current status
    current_status = "Unknown"
    if "occasional backache" in transcript_text.lower():
        current_status = "Occasional backache"
    elif "better" in transcript_text.lower():
        current_status = "Improving"
    
    # Determine prognosis
    prognosis = "Unknown"
    if "improving" in transcript_text.lower():
        prognosis = "Improving, full recovery expected"
    
    return {
        "Patient_Name": patient_name,
        "Symptoms": medical_details["Symptoms"],
        "Diagnosis": medical_details["Diagnosis"][0] if medical_details["Diagnosis"] else "Not specified",
        "Treatment": medical_details["Treatment"],
        "Current_Status": current_status,
        "Prognosis": prognosis
    }

# Sentiment & Intent Analysis - adapted from physician_notetaker.ipynb
def analyze_sentiment_and_intent(patient_text):
    # Extract patient statements
    patient_statements = []
    for line in patient_text.split('\n'):
        if line.strip().startswith("Patient:"):
            patient_statements.append(line.strip()[8:].strip())
    
    patient_combined = " ".join(patient_statements)
    
    # Sentiment analysis using the transformer model
    classification = st.session_state.sentiment_model(patient_combined)
    raw_label = classification[0]['label']
    
    # Map raw sentiment label to our desired format
    if raw_label == 'POSITIVE':
        sentiment = "Reassured"
    elif raw_label == 'NEGATIVE':
        sentiment = "Anxious"
    else:
        sentiment = "Neutral"
    
    # Rule-based intent detection from your notebook
    lowered_text = patient_combined.lower()
    if "worry" in lowered_text or "anxious" in lowered_text or "concern" in lowered_text:
        intent = "Seeking reassurance"
    elif "better" in lowered_text or "improving" in lowered_text or "helped" in lowered_text:
        intent = "Reporting improvement"
    elif "pain" in lowered_text or "symptom" in lowered_text:
        intent = "Reporting symptoms"
    else:
        intent = "Providing information"
    
    return {
        "Sentiment": sentiment,
        "Intent": intent,
        "Confidence": classification[0]['score']
    }

# SOAP Note Generation - adapted from physician_notetaker.ipynb
def generate_soap_note(summary, transcript_text):
    # Extract more context for better SOAP note
    history = "Unknown"
    if "car accident" in transcript_text.lower():
        history = "Patient involved in a car accident"
        if "September" in transcript_text:
            history += " in September"
    
    physical_exam = "Unknown"
    if "physical examination" in transcript_text.lower():
        physical_exam = "Physical examination mentioned, details not provided"
    
    assessment = "Unknown"
    if summary["Diagnosis"] != "Not specified":
        assessment = summary["Diagnosis"]
    
    plan = "Unknown"
    if "physiotherapy" in transcript_text.lower():
        plan = "Continue physiotherapy"
    if "painkiller" in transcript_text.lower():
        plan += ", use painkillers as needed"
    
    return {
        "Subjective": {
            "Chief_Complaint": ", ".join(summary["Symptoms"]),
            "History_of_Present_Illness": history
        },
        "Objective": {
            "Physical_Exam": physical_exam,
            "Observations": "Based on patient's statements"
        },
        "Assessment": {
            "Diagnosis": summary["Diagnosis"],
            "Severity": "Improving based on patient statements"
        },
        "Plan": {
            "Treatment": plan,
            "Follow_Up": "As needed based on symptom progression"
        }
    }

# Main application logic
tab1, tab2, tab3 = st.tabs(["📋 Transcript Analysis", "📊 Batch Processing", "ℹ️ About"])

with tab1:
    # Input section
    st.markdown("<div class='subheader-text'>Input Medical Transcript</div>", unsafe_allow_html=True)
    
    # Sample transcript option - this is the exact transcript from your notebook
    sample_transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when 
I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain 
in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but 
they didn't do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take 
painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help 
with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty 
concentrating?
Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from 
doing anything.
Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
"""
    
    use_sample = st.checkbox("Use sample transcript", value=True)
    
    if use_sample:
        transcript = st.text_area("Medical Transcript", value=sample_transcript, height=300)
    else:
        transcript = st.text_area("Medical Transcript", height=300, 
                                placeholder="Enter physician-patient conversation...")
    
    # Load models when needed
    if not st.session_state.models_loaded and (transcript.strip() != ""):
        load_models()
    
    # Process transcript button
    analyze_button = st.button("Analyze Transcript", type="primary", disabled=not st.session_state.models_loaded)
    
    if analyze_button and transcript:
        with st.spinner('Analyzing transcript...'):
            # Run the analysis pipeline - this calls the functions that mirror your notebook
            medical_details = extract_medical_details(transcript)
            summary = structured_summary(medical_details, transcript)
            sentiment_analysis = analyze_sentiment_and_intent(transcript)
            soap_note = generate_soap_note(summary, transcript)
            
            # Display results
            st.markdown("<div class='subheader-text'>Analysis Results</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Medical Details")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write("**Symptoms:**", ", ".join(medical_details["Symptoms"]) if medical_details["Symptoms"] else "None detected")
                st.write("**Treatments:**", ", ".join(medical_details["Treatment"]) if medical_details["Treatment"] else "None detected")
                st.write("**Diagnosis:**", ", ".join(medical_details["Diagnosis"]) if medical_details["Diagnosis"] else "None detected")
                st.write("**Timeframes:**", ", ".join(medical_details["Timeframes"]) if medical_details["Timeframes"] else "None detected")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 📊 Sentiment Analysis")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write("**Patient Sentiment:**", sentiment_analysis["Sentiment"])
                st.write("**Patient Intent:**", sentiment_analysis["Intent"])
                st.write("**Confidence Score:**", f"{sentiment_analysis['Confidence']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📝 Patient Summary")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                for key, value in summary.items():
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ')}:**", ", ".join(value) if value else "Not specified")
                    else:
                        st.write(f"**{key.replace('_', ' ')}:**", value)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 📑 SOAP Note")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                
                # Subjective
                st.markdown("**S - Subjective**")
                st.write("Chief Complaint:", soap_note["Subjective"]["Chief_Complaint"])
                st.write("History of Present Illness:", soap_note["Subjective"]["History_of_Present_Illness"])
                
                # Objective
                st.markdown("**O - Objective**")
                st.write("Physical Exam:", soap_note["Objective"]["Physical_Exam"])
                st.write("Observations:", soap_note["Objective"]["Observations"])
                
                # Assessment
                st.markdown("**A - Assessment**")
                st.write("Diagnosis:", soap_note["Assessment"]["Diagnosis"])
                st.write("Severity:", soap_note["Assessment"]["Severity"])
                
                # Plan
                st.markdown("**P - Plan**")
                st.write("Treatment:", soap_note["Plan"]["Treatment"])
                st.write("Follow-Up:", soap_note["Plan"]["Follow_Up"])
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Option to download results as JSON
            result_json = {
                "medical_details": medical_details,
                "summary": summary,
                "sentiment_analysis": sentiment_analysis,
                "soap_note": soap_note
            }
            
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(result_json, indent=4),
                file_name="medical_analysis_results.json",
                mime="application/json",
            )

with tab2:
    st.markdown("<div class='subheader-text'>Batch Processing</div>", unsafe_allow_html=True)
    st.markdown("""
    Upload multiple transcripts for batch processing. Each transcript should be a separate text file.
    """)
    
    uploaded_files = st.file_uploader("Upload transcript files", accept_multiple_files=True, type=['txt'])
    
    if uploaded_files:
        if not st.session_state.models_loaded:
            load_models()
        
        process_batch = st.button("Process Batch", type="primary", disabled=not st.session_state.models_loaded)
        
        if process_batch:
            with st.spinner('Processing files...'):
                results = []
                
                for uploaded_file in uploaded_files:
                    # Read file content
                    content = uploaded_file.read().decode("utf-8")
                    
                    # Extract filename without extension
                    filename = uploaded_file.name.split('.')[0]
                    
                    # Run analysis pipeline (reusing functions from Tab 1)
                    medical_details = extract_medical_details(content)
                    summary = structured_summary(medical_details, content)
                    sentiment_analysis = analyze_sentiment_and_intent(content)
                    soap_note = generate_soap_note(summary, content)
                    
                    # Compile results
                    results.append({
                        "filename": filename,
                        "medical_details": medical_details,
                        "summary": summary,
                        "sentiment_analysis": sentiment_analysis,
                        "soap_note": soap_note
                    })
                
                # Display batch results
                st.success(f"Processed {len(results)} files successfully!")
                
                # Create a DataFrame for display
                results_df = pd.DataFrame({
                    "Filename": [r["filename"] for r in results],
                    "Patient": [r["summary"]["Patient_Name"] for r in results],
                    "Diagnosis": [r["summary"]["Diagnosis"] for r in results],
                    "Sentiment": [r["sentiment_analysis"]["Sentiment"] for r in results],
                    "Current Status": [r["summary"]["Current_Status"] for r in results]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Option to download batch results
                st.download_button(
                    label="Download Batch Results as JSON",
                    data=json.dumps(results, indent=4),
                    file_name="batch_analysis_results.json",
                    mime="application/json",
                )

with tab3:
    st.markdown("<div class='subheader-text'>About This Application</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Medical Transcript Analyzer
    
    This application uses Natural Language Processing (NLP) techniques to analyze medical conversation transcripts 
    and automatically generate structured information that can assist healthcare professionals.
    
    #### Features:
    - **Medical Entity Extraction**: Identifies symptoms, treatments, diagnoses, and timeframes
    - **Sentiment Analysis**: Determines patient emotional state and communicative intent
    - **Structured Summarization**: Creates a concise patient summary
    - **SOAP Note Generation**: Produces a standardized clinical note format
    
    #### Technologies Used:
    - **SpaCy**: For named entity recognition and linguistic analysis
    - **Transformers (HuggingFace)**: For sentiment analysis
    - **Streamlit**: For the web application interface
    
    #### Methods and Algorithms:
    - Named Entity Recognition (NER) with rule-based extensions
    - Transformer-based sentiment classification
    - Rule-based intent detection
    - Structured information extraction through pattern matching
    
    #### Developed by:
    Shiva Sai
    
    GitHub: [shiva-sai-824](https://github.com/shiva-sai-824)
    """)
    
    # Add a GitHub link button
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 2rem;">
        <a href="https://github.com/shiva-sai-824/physician-notetaker" target="_blank" style="text-decoration: none;">
            <button style="background-color: #333; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; border: none; display: flex; align-items: center; font-weight: bold;">
                <svg height="24" width="24" viewBox="0 0 16 16" version="1.1" style="margin-right: 0.5rem;">
                    <path fill="white" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
                View Project on GitHub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Add a footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 1rem;">
    Medical Transcript Analyzer | Developed for Physician Note-Taking Applications
</div>
""", unsafe_allow_html=True)

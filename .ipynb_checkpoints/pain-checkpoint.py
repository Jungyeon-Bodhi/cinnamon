#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:00:23 2025

@author: ijeong-yeon
"""

import google.generativeai as genai
import os
import time
import PyPDF2
import pandas as pd

"""
[1] Set-up Google AI
"""
def setup_AI(api):
    GOOGLE_API_KEY = api
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Set up the model
    generation_config = {
      "temperature": 0.9,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048,
    }
    
    model = genai.GenerativeModel('gemini-2.0-flash',generation_config=generation_config)
    return model

"""
[2] Extracting ASEAN policy documents
"""
def read_docs(file_path):
    data_folder = file_path  
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    pdf_files = sorted(pdf_files, key=lambda x: int(os.path.splitext(x)[0]))
    pdf_numbers = sorted([int(os.path.splitext(f)[0]) for f in pdf_files])
    num_pdfs = len(pdf_files)
    print(f"A total of {num_pdfs} PDF files exist.")
    docs = {}
    for i, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(data_folder, pdf_file)
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            docs[f"doc{i}"] = text
    return docs, pdf_numbers

"""
[3] Generating responses
"""
def generate_ai(model, pdf_numbers, docs, file_name):
    results = []
    for doc, i in zip(list(docs.keys()), pdf_numbers):
        prompt = f"""
        {docs[doc]} <- This is the policy document, please answer the following six questions with the score and the reason behind (just a bit strict, transformation and potential for transformation are different).
1) Question 1: To what extent does the policy/project address the intended challenge/issue? - Question 1 Criteria is as followed = Score 1 (Poorly) - Does not state the challenge it intends to address/unclear what challenge it intends to address (towards a increased understanding, tolerance and a sense of regional agenda's), Score 2 (Somewhat) - States the challenge it aims to address but not why addressing this matters? (towards a increased understanding, tolerance and a sense of regional agenda's), Score 3 (Greatly) - States what the challenge is clearly and why this challenge is important to fostering a sense of increased understanding, tolerance and a sense of regional agenda's.
2) Question 2: To what extent is the policy/project contextually appropriate? - Question 2 Criteria is as followed = Score 1 (poorly) - The project appears at odds with the contextual dynamics of the time it was set up/ The project makes no adaptations to its contextual dynamics. Score 2 (Somewhat) - The project appears to have made some adaptations towards context but could do with improvement/ context has changed since project inception and it has not adapted to new context. Score 3 (greatly) - The project is contextually relevant.
3) Question 3: To what extent was the policy/project designed through a consultative process and include meaningful feedback opportunities? - Question 3 Criteria is as followed = Score 1 (poorly) The policy/project faces significant opposition from most or all stakeholders. Or there is a significant opposition to it. Score 2 (somewhat) - The policy/project is mostly supported by key stakeholders with minor opposition. Score 3 (Greatly) - The policy/project enjoys broad support from all or nearly all key stakeholder groups.
4) Question 4: To what extent Is the policy/project widely accepted among key stakeholder groups? - Question 4 Criteria is as followed = Score 1 (Poorly) - No signs of acceptance/ outright denied acceptance. Score 2 (somewhat) - Signs of some acceptance. Score 3 (greatly) - Sign of total or majority acceptance.
5) Question 5: Is there an operational Monitoring and Evaluation (M&E) framework in place, and to what extent is it adhered to? - Question 5 Criteria is as followed = Score 1 (poorly) - No M&E framework to be found. Score 2 (somewhat) - Appears to be some attempt or mention of  M&E. Score 3 (greatly) - M&E framework intact.
6) Question 6: To what extent does the policy/project have transformative potential? - Question 6 Criteria is as followed = Score 1 (poorly) The policy/project introduces minor or incremental changes with no significant disruption to existing systems/It lacks scalability and is unlikely to inspire further innovation/ Short-term focus and impacts. Score 2 (somewhat) - The policy/project brings noticeable improvements but does not fundamentally alter existing structures/ It has some scalability, but implementation may face barriers. Score 3 (greatly) - The policy/project introduces groundbreaking or system-wide changes, reshaping existing frameworks/ leads to long-term attitudinal, behavioural or policy changes.
"""    
        response = model.generate_content(prompt)
        temp = []
        results.append({'Document': f'{i}.pdf','Response': response.text})
        temp.append({'Document': f'{i}.pdf','Response': response.text})
        temp_df = pd.DataFrame(temp)
        temp_df.to_excel(f"data/generated/{i}.xlsx")
        print(f"Finished: {i}.xlsx")
        time.sleep(70)
    results_df = pd.DataFrame(results)
    results_df.to_excel(f"data/{file_name}.xlsx")
    return results_df

"""
[4] Pipeline
"""
api = 'AIzaSyBwjlLmM1X5brUWVgUkA17GkQuFje8AIXs'
file_path = "data/"
doc_name = "ASEAN_effectiveness_assessment"

cinnamon = setup_AI(api)
docs, pdf_numbers = read_docs(file_path)
df = generate_ai(cinnamon, pdf_numbers, docs, doc_name)
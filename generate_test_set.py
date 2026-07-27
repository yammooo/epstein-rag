import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-2.5-flash')

df = pd.read_parquet("artifacts/raw_dataset.parquet")
df_valid = df[df["body"].str.len() > 200]
sample_df = df_valid.sample(15)

test_set = []
print("Generating 15 questions... This will take about 3-4 minutes to respect rate limits.")

for idx, row in sample_df.iterrows():
    success = False
    retries = 5
    while not success and retries > 0:
        try:
            sender = row.get("from_name", "Unknown")
            rec = row.get("to", "Unknown")
            date = row.get("date", "Unknown")
            subj = row.get("subject", "Unknown")
            
            context = f"From: {sender}\nTo: {rec}\nDate: {date}\nSubject: {subj}\n\nBody: {row['body']}"
            prompt = f"Based on the following email passage, generate one complex question and its correct answer. Return JSON with keys 'question' and 'answer'.\n\nEMAIL PASSAGE:\n{context}"
            
            response = model.generate_content(prompt)
            json_str = response.text.strip().replace('```json', '').replace('```', '')
            qa = json.loads(json_str)
            qa["source_doc_ids"] = [row["doc_id"]]
            test_set.append(qa)
            print(f"Generated Q{len(test_set)}")
            success = True
            time.sleep(15) # Wait 15 seconds to stay well under 5 per minute
        except Exception as e:
            print(f"Rate limit hit, waiting 30 seconds... (Error: {e})")
            time.sleep(30)
            retries -= 1

with open("artifacts/micro_test_set.json", "w") as f:
    json.dump(test_set, f, indent=2)

print(f"Saved {len(test_set)} QA pairs to artifacts/micro_test_set.json")

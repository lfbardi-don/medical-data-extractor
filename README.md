# Medical Data Extractor

A simple Python tool that turns free‑text medical notes into structured patient data.

---

## 🔍 What It Does

1. **Extracts Age & Treatment**\
   Uses OpenAI’s function‑calling API to identify the patient’s age and the recommended treatment or procedure from each transcription.

2. **Looks Up ICD Codes**\
   Automatically fetches the official ICD codes for any treatments found.

3. **Retries & Caches**

   - Retries on temporary errors.
   - Caches repeated ICD lookups for faster results.

---

## 🛠️ Prerequisites

- **Python 3.8+**
- An **OpenAI API key** (set in environment)
- A **CSV file** with at least two columns:
  - `medical_specialty` (e.g., `cardiology`)
  - `transcription` (the raw text to process)

---

## 🚀 Getting Started

1. **Clone this repo**

   ```bash
   git clone https://github.com/<your‑username>/medical-data-extractor.git
   cd medical-data-extractor
   ```

2. **Set up your environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # .venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Configure** Create a file named `.env` in the project root with these entries:

   ```ini
   OPENAI_API_KEY=sk-XXXXXXX
   OPENAI_MODEL=gpt-3.5-turbo     # optional, defaults to gpt-3.5-turbo
   INPUT_CSV_PATH=data/transcriptions.csv
   OUTPUT_CSV_PATH=data/structured_medical_data.csv
   ```

4. **Prepare your input** Place your `transcriptions.csv` under `data/`. It should look like:

   ```csv
   medical_specialty,transcription
   cardiology,"54-year-old male with chest pain..."
   dermatology,"28-year-old female with rash on arms..."
   ```

5. **Run the extractor**

   ```bash
   python medical_data_extractor.py
   ```

   - Reads from `INPUT_CSV_PATH`
   - Writes structured data to `OUTPUT_CSV_PATH`

6. **Check the result** Open `data/structured_medical_data.csv`. You’ll see columns:

   - `age`
   - `recommended_treatment`
   - `icd_codes` (a JSON array of codes)
   - `medical_specialty`

---

## 💡 Usage Example in Code

You can also import and run it directly in your Python projects:

```python
from medical_data_extractor import MedicalExtractor

extractor = MedicalExtractor()
df = extractor.run("data/transcriptions.csv")
print(df.head())
```

---

## 🤝 Contributing

Feel free to open an issue or submit a pull request. Suggestions are welcome.

---

import json
import logging
from functools import lru_cache
from typing import List, Dict, Optional

import pandas as pd
from decouple import config
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MedicalExtractor:
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        icd_temperature: float = 0.3
    ):
        # Load configuration
        self.api_key = api_key or config("OPENAI_API_KEY")
        self.model = model or config("OPENAI_MODEL", default="gpt-3.5-turbo")
        self.icd_temperature = icd_temperature

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Function schema for extracting medical data
        self.functions = [
            {
                "name": "extract_medical_data",
                "description": "Extracts patient's age and recommended treatment from transcription.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "description": "Age of the patient"},
                        "recommended_treatment": {"type": "string", "description": "Recommended treatment or procedure"}
                    },
                    "required": ["age", "recommended_treatment"]
                }
            }
        ]

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def extract_info(self, transcription: str) -> Dict:
        """
        Calls OpenAI to extract age and recommended treatment from a transcription.
        Retries up to 3 times on failure with exponential backoff.
        """
        # Validate input
        if not transcription:
            logger.warning("Empty transcription received, skipping extraction.")
            return {"age": None, "recommended_treatment": "Unknown"}

        messages = [
            {"role": "system", "content": "You are a healthcare professional extracting patient data. Always return both age and recommended treatment. If missing, use 'Unknown'."},
            {"role": "user", "content": f"Please extract the patient's age and recommended treatment from the following transcription: {transcription}"}
        ]

        logger.debug("Sending extraction request to OpenAI for transcription: %s", transcription[:50])
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=self.functions,
            function_call="auto"
        )

        fc = response.choices[0].message.function_call
        args = json.loads(fc.arguments)
        age = args.get("age")
        treatment = args.get("recommended_treatment", "Unknown")
        logger.info("Extracted age=%s, treatment=%s", age, treatment)

        return {"age": age, "recommended_treatment": treatment}

    @lru_cache(maxsize=256)
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def get_icd_codes(self, treatment: str) -> List[str]:
        """
        Retrieves ICD codes for a given treatment via OpenAI.
        Caches results to avoid redundant calls.
        """
        if not treatment or treatment.lower() == "unknown":
            return []

        prompt = (
            f"Provide the ICD codes for the following treatment or procedure: '{treatment}'. "
            "Return the answer as a JSON array of code strings, with no additional text."
        )
        logger.debug("Requesting ICD codes for treatment: %s", treatment)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.icd_temperature
        )

        try:
            codes = json.loads(response.choices[0].message.content)
            logger.info("Retrieved ICD codes for '%s': %s", treatment, codes)
            return codes
        except json.JSONDecodeError:
            logger.error("Failed to parse ICD codes JSON. Response: %s", response.choices[0].message.content)
            return []

    def run(
        self,
        input_csv: str,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads transcriptions from input_csv, processes each row,
        and returns a DataFrame with structured data.
        Optionally saves to output_csv.
        """
        logger.info("Loading data from %s", input_csv)
        df = pd.read_csv(input_csv)

        def process_row(row):
            result = self.extract_info(row.get("transcription", ""))
            icd_codes = self.get_icd_codes(result.get("recommended_treatment", "Unknown"))
            return {
                "age": result.get("age"),
                "recommended_treatment": result.get("recommended_treatment"),
                "icd_codes": icd_codes,
                "medical_specialty": row.get("medical_specialty")
            }

        logger.info("Processing %d rows", len(df))
        df_structured = df.apply(process_row, axis=1, result_type="expand")

        if output_csv:
            logger.info("Saving structured data to %s", output_csv)
            df_structured.to_csv(output_csv, index=False)

        return df_structured


if __name__ == "__main__":
    extractor = MedicalExtractor()
    # Paths can also be provided via environment or CLI in a future enhancement
    in_path = config("INPUT_CSV_PATH", default="data/transcriptions.csv")
    out_path = config("OUTPUT_CSV_PATH", default="data/structured_medical_data.csv")
    extractor.run(in_path, out_path)

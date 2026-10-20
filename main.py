
import json
import os
import sys
import time
from datetime import date

from dotenv import load_dotenv
from google import genai
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


load_dotenv()

PROMOTION_TEXT = (
    "Ariel 3-in-1 Pods will be on promotion at Tesco and Asda from the first "
    "of April through to the thirtieth, offering shoppers a twenty percent "
    "saving off the standard shelf price. This promotion does not apply to "
    "convenience stores."
)

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_RETRIES = 3  # Three retries means four total attempts.


class Promotion(BaseModel):
    """Validated fields extracted from a product promotion."""

    model_config = ConfigDict(strict=True, extra="forbid")

    product_name: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    discount_percentage: int = Field(ge=0, le=100)
    promotion_start_date: date
    promotion_end_date: date
    eligible_retailers: list[str] = Field(min_length=1)

    @field_validator(
        "promotion_start_date",
        "promotion_end_date",
        mode="before",
    )
    @classmethod
    def parse_iso_dates(cls, value):
        """Parse dates supplied as YYYY-MM-DD strings."""
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value


def build_prompt(original_text: str) -> str:
    """Ask the model to extract the promotion details as JSON."""

    return f"""
Extract the promotion details from the text below.

Return only a JSON object with exactly these fields:
- product_name: string (product name excluding the brand)
- brand: string
- discount_percentage: integer from 0 to 100
- promotion_start_date: date in YYYY-MM-DD format
- promotion_end_date: date in YYYY-MM-DD format
- eligible_retailers: list of strings

Rules:
- Use only information supported by the text.
- Do not include excluded retailers or store types.
- The text does not specify a year. Use 2027 for the April dates
  for this exercise.
- Do not include Markdown fences or explanatory text.

Promotion text:
{original_text}
""".strip()


def extract_promotion(client, original_text: str) -> Promotion:
    """Extract, validate, and retry failed model responses."""

    prompt = build_prompt(original_text)
    last_error = "No response received."

    for attempt in range(MAX_RETRIES + 1):
        raw_output = "No response received."

        print(
            f"Sending promotion text to Gemini... "
            f"Attempt {attempt + 1}/{MAX_RETRIES + 1}",
            flush=True,
        )

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            raw_output = response.text or ""
            if not raw_output.strip():
                raise ValueError("The model returned an empty response.")

            data = json.loads(raw_output)
            promotion = Promotion.model_validate(data)

            if promotion.promotion_end_date < promotion.promotion_start_date:
                raise ValueError(
                    "Promotion end date must not precede start date."
                )

            if promotion.promotion_start_date.month != 4:
                raise ValueError("Promotion start date must be in April.")

            if promotion.promotion_end_date.month != 4:
                raise ValueError("Promotion end date must be in April.")

            return promotion

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = f"Response validation failed: {exc}"

            if attempt == MAX_RETRIES:
                break

            prompt = f"""
Your previous response failed validation. Correct the response.

Original instructions:
{build_prompt(original_text)}

Previous response:
{raw_output}

Validation error:
{last_error}

Return only a corrected JSON object with the exact required fields
and types. Do not include Markdown fences or explanatory text.
""".strip()

            print("Invalid response. Retrying with validation feedback.", flush=True)

        except Exception as exc:
            error_message = str(exc)
            status_code = (
                getattr(exc, "code", None)
                or getattr(exc, "status_code", None)
            )

            retryable = (
                status_code in (408, 429, 500, 502, 503, 504)
                or any(
                    marker in error_message.upper()
                    for marker in (
                        "408",
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "RESOURCE_EXHAUSTED",
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                        "INTERNAL",
                        "TIMEOUT",
                    )
                )
            )

            if not retryable or attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Gemini API request failed: {exc}"
                ) from exc

            last_error = f"Temporary Gemini API error: {exc}"
            delay = min(2 ** (attempt + 1), 8)

            print(
                f"Temporary API error. Retrying in {delay} seconds...",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Could not extract valid promotion data after "
        f"{MAX_RETRIES + 1} attempts. Last error: {last_error}"
    )


def main() -> int:
    """Run the extraction and print validated JSON."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY is missing. Check your local .env file.",
            file=sys.stderr,
        )
        return 1

    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 60000},
        )

        promotion = extract_promotion(client, PROMOTION_TEXT)
        print(promotion.model_dump_json(indent=2))
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

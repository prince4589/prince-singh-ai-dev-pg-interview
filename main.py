
import json
import os
import sys
from datetime import date
from typing import List

from google import genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROMOTION_TEXT = (
    "Ariel 3-in-1 Pods will be on promotion at Tesco and Asda from the first "
    "of April through to the thirtieth, offering shoppers a twenty percent "
    "saving off the standard shelf price. This promotion does not apply to "
    "convenience stores."
)

MAX_RETRIES = 3


class Promotion(BaseModel):
    """Validated fields extracted from a product promotion."""

    model_config = ConfigDict(strict=True, extra="forbid")

    product_name: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    discount_percentage: int = Field(ge=0, le=100)
    promotion_start_date: date
    promotion_end_date: date
    eligible_retailers: List[str] = Field(min_length=1)


def build_prompt(original_text: str) -> str:
    """Ask the LLM to extract the promotion fields as JSON."""

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
    """Extract data, validate it, and retry invalid responses."""

    prompt = build_prompt(original_text)
    last_error = "No response received."

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                },
            )

            raw_output = response.text
            if not raw_output:
                raise ValueError("The model returned an empty response.")

            data = json.loads(raw_output)
            promotion = Promotion.model_validate(data)

            if promotion.promotion_end_date < promotion.promotion_start_date:
                raise ValueError(
                    "Promotion end date must not precede start date."
                )

            return promotion

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc)

            if attempt == MAX_RETRIES:
                break

            prompt = f"""
Your previous response failed validation.

Original promotion text:
{original_text}

Previous response:
{raw_output if 'raw_output' in locals() else 'No valid response received'}

Validation error:
{last_error}

Please correct the response. Return only a JSON object with the exact
required fields and types from the original instructions.
""".strip()

        except Exception as exc:
            # API errors are not necessarily fixed by repeating the same request.
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    raise RuntimeError(
        f"Could not extract valid promotion data after "
        f"{MAX_RETRIES + 1} attempts. Last error: {last_error}"
    )


def main() -> int:
    """Run extraction and print the validated JSON."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY is not set. Configure your Gemini API key.",
            file=sys.stderr,
        )
        return 1

    try:
        client = genai.Client(api_key=api_key)
        promotion = extract_promotion(client, PROMOTION_TEXT)
        print(promotion.model_dump_json(indent=2))
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

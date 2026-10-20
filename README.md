# P&G AI Development Technical Task

## Overview

This project extracts structured promotion data from a natural-language product promotion using a Large Language Model (LLM).

The script will send the input text to the Google Gemini API, validate the response against a defined schema, and retry with corrective feedback if the response is invalid.

## Requirements

- Python 3.10+
- Google Gemini API key
- Google Gen AI Python SDK
- Pydantic for data validation

## Planned Approach

1. Define the input promotion text and the expected output schema.
2. Send a prompt to Gemini requesting the required fields as JSON.
3. Validate the response using Pydantic, including field types and ISO 8601 dates.
4. If parsing or validation fails, retry with a follow-up prompt containing the original text, failed response, and validation errors.
5. Allow a maximum of three retries after the initial attempt.
6. Print the validated JSON on success, or a clear error message if all attempts fail.

## Schema

- `product_name`: string
- `brand`: string
- `discount_percentage`: integer
- `promotion_start_date`: date in `YYYY-MM-DD` format
- `promotion_end_date`: date in `YYYY-MM-DD` format
- `eligible_retailers`: list of strings

## Error Handling

The implementation will distinguish between invalid JSON, schema validation failures, and API errors. API credentials will be loaded from an environment variable rather than hardcoded in the source code.

## Assumptions

The supplied promotion does not specify a year. The implementation will document its date interpretation explicitly rather than silently treating an inferred year as a fact.

The promotion excludes convenience stores, which should not be included in the eligible retailer list.

## Testing Plan

- Validate a successful extraction.
- Reject missing fields and incorrect field types.
- Reject invalid date formats and impossible dates.
- Verify that malformed responses trigger retries.
- Verify that the script stops after the retry limit and reports failure clearly.

## Future Improvements

With more time, I would add automated tests with mocked API responses, structured logging, API rate-limit handling, and additional tests for ambiguous promotion details.

## Status

- [x] Initial project plan documented
- [ ] Repository structure and dependencies
- [ ] Gemini integration
- [ ] Schema validation
- [ ] Retry and error handling
- [ ] Automated tests
- [ ] End-to-end verification

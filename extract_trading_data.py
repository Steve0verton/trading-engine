import os
import base64
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from anthropic import Anthropic, AnthropicError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration for the table extractor."""

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    MIME_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    DEFAULT_MIME_TYPE = "image/png"
    DEFAULT_INPUT_FOLDER_NAME = "input"
    OUTPUT_FILE_NAME = "consolidated_trading_data.csv"
    CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
    MAX_TOKENS = 4000


class TableExtractionError(Exception):
    """Custom exception for table extraction errors."""

    pass


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension."""
    ext = Path(image_path).suffix.lower()
    return Config.MIME_TYPES.get(ext, Config.DEFAULT_MIME_TYPE)


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise TableExtractionError(f"Failed to read image {image_path}: {str(e)}")


def get_claude_prompt() -> str:
    """Return the Claude prompt for table extraction."""
    return """
    This image contains a stock market trading table with multiple columns.

    Analyze this table and extract the following data for EACH ticker symbol:
    1. The section/category (like "EMERGING MARKETS", "US MAIN ETFs", "COMMODITIES", etc.)
    2. Ticker symbol
    3. Percentage change value
    4. MOM SCORE value
    5. IV OTM value (green C/P or red)
    6. IV RANK SCORE value
    7. SHORT TERM STRUCTURE SIGNAL value
    8. UNUSUAL OPTION value
    9. FAIR VALUE GAP value
    10. POTENTIAL SHIFT value
    11. INTRADAY FLOW value
    12. PUT/CALL volume ratio value
    13. ORDER BLOCK FLOW value
    14. NET 3D FLOW CHANGE value

    Return ONLY valid JSON with this exact structure:
    {
      "data": [
        {
          "section": "CATEGORY_NAME",
          "ticker": "SYMBOL",
          "change_pct": "VALUE",
          "mom_score": "VALUE",
          "iv_otm": "VALUE",
          "iv_rank": "VALUE",
          "short_term_signal": "VALUE",
          "unusual_option": "VALUE",
          "fair_value_gap": "VALUE",
          "potential_shift": "VALUE",
          "intraday_flow": "VALUE",
          "put_call_ratio": "VALUE",
          "order_block": "VALUE",
          "net_3d_change": "VALUE"
        }
      ]
    }
    """


def extract_table_from_image(client: Anthropic, image_path: str) -> Optional[Dict]:
    """
    Extract trading table data from an image using Claude API.

    Args:
        client: Anthropic API client
        image_path: Path to the image file

    Returns:
        Extracted data in JSON format or None if extraction fails
    """
    try:
        image_base64 = image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        logger.info(f"Processing image: {Path(image_path).name} (MIME: {mime_type})")

        message = client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=Config.MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": get_claude_prompt()},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_base64,
                            },
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            logger.error(f"No JSON found in response for {Path(image_path).name}")
            return None

        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)

    except (json.JSONDecodeError, AnthropicError) as e:
        logger.error(f"Failed to process {Path(image_path).name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {Path(image_path).name}: {str(e)}")
        return None


def get_image_paths(folder_path: str) -> List[str]:
    """Get list of supported image file paths from folder."""
    image_paths = [
        str(Path(folder_path) / file)
        for file in os.listdir(folder_path)
        if Path(file).suffix.lower() in Config.SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def process_images_from_folder(api_key: str, folder_path: str) -> pd.DataFrame:
    """
    Process all images in folder and extract trading data.

    Args:
        api_key: Anthropic API key
        folder_path: Path to folder containing images

    Returns:
        DataFrame with consolidated data
    """
    try:
        client = Anthropic(api_key=api_key)
    except AnthropicError as e:
        logger.error(f"Failed to initialize Anthropic client: {str(e)}")
        return pd.DataFrame()

    image_paths = get_image_paths(folder_path)
    if not image_paths:
        logger.warning(f"No images found in {folder_path}")
        return pd.DataFrame()

    logger.info(f"Found {len(image_paths)} images to process")
    all_data = []

    for image_path in image_paths:
        extracted_data = extract_table_from_image(client, image_path)
        if extracted_data and "data" in extracted_data:
            all_data.extend(extracted_data["data"])
            logger.info(
                f"Extracted {len(extracted_data['data'])} tickers from {Path(image_path).name}"
            )
        else:
            logger.warning(f"No valid data extracted from {Path(image_path).name}")

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def parse_arguments() -> str:
    """Parse command-line arguments for input folder."""
    parser = argparse.ArgumentParser(description="Stock Market Trading Table Extractor")
    parser.add_argument(
        "--input-folder",
        type=str,
        default=Config.DEFAULT_INPUT_FOLDER_NAME,
        help="Folder containing input images (default: input)",
    )
    args = parser.parse_args()
    return args.input_folder


def main():
    """Main function to run the table extraction process."""
    logger.info("Starting Stock Market Trading Table Extractor")

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Please enter your Anthropic API key: ")

    # Get input folder from arguments
    input_folder_name = parse_arguments()
    script_dir = Path(__file__).parent
    input_folder = script_dir / input_folder_name

    # Create input folder if it doesn't exist
    input_folder.mkdir(exist_ok=True)
    if not any(input_folder.iterdir()):
        logger.warning(f"No files found in {input_folder}. Please add images.")
        return

    # Process images
    result_df = process_images_from_folder(api_key, str(input_folder))

    # Save results
    if not result_df.empty:
        output_path = script_dir / Config.OUTPUT_FILE_NAME
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(result_df)} rows to {output_path}")
        logger.info("Sample extracted data:\n%s", result_df.head().to_string())
    else:
        logger.error("No data extracted. Please verify image content.")


if __name__ == "__main__":
    main()

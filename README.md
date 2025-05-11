# Trading Engine

Trading engine which assimilates data across various sources to provide automated analysis and trade ideas.

## How to Use

* Install dependencies: `pip install pandas anthropic`
* Set API key: `export ANTHROPIC_API_KEY='your-key'` (or enter when prompted)
* Put the image (or images) in `input` directory
  * Make sure `input` directory exists: `mkdir -p input`
* Default usage: `python extract_trading_data.py`
* To use a custom input directory: `python extract_trading_data.py --input-folder your_folder_name`

## Output

* Creates `consolidated_trading_data.csv` in the root directory where `extract_trading_data.py` lives
* Shows processing logs in the console

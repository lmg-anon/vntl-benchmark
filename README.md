# vntl-benchmark
Visual Novel Translation Benchmark

This script evaluates the ability of Large Language Models (LLMs) to translate Japanese visual novels into English. The leaderboard generated using this script can be found [here](https://huggingface.co/datasets/lmg-anon/vntl-leaderboard).

## Getting Started

1. Create a virtual environment (optional) and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. Download the datasets.  
   *(Currently not readily available)*

4. Rename the file `config.example.yml` to `config.yml`.

5. Choose or create a model configuration:
   - Use one of the existing file names in the "configs" folder, or
   - Create a new file in the "configs" folder with the name "org@model#quant.yml" or "org@model.yml" if it's a cloud model. For examples, see the other files.
   
   Note: You can configure the backend either in `config.yml` (in the "custom_backends" part) or in the specific config folder (in the "backend" part). The latter takes precedence.

6. Run the benchmark:
   ```
   python runner.py --model org@model#quant --dataset Senren
   python runner.py --model org@model#quant --dataset Mashiro
   ```

If everything has gone right, the results will be generated.

This script is still in development. A more comprehensive README will be provided soon™.

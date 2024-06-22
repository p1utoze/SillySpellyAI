# SillySpellyAI
    🔤 A friendly simple spell checker app powered by AI and built with Python. Made in a hackathon within 3 hours.

## Features
- ✍ Highlights misspelled words
- 🔨 Auto-corrects misspelled words
- 🤔 Reasons why a word in the sentence is misspelled
- 🔊 An audio feature that reads out the corrected sentence
![app.png](assets%2Fapp.png)
## Installation
1. Clone the repository
```bash
git clone https://github.com/p1utoze/SillySpellyAI.git
```

2. Change directory to the project folder
```bash
cd SillySpellyAI
```

3. Create a virtual environment
```bash
python -m venv venv
```

4. Activate the virtual environment
```bash
# For Windows
venv\Scripts\activate

# For MacOS and Linux
source venv/bin/activate
```

## Usage

#### Running locally on your machine

1. Install the required packages
```bash
pip install -r requirements.txt
```

2. Get the API token from https://modal.com/ and set it from the CLI
```bash
modal token set --token-id <token_id> --token-secret <token_secret>
```

3. Run the LLama API server in the background
```bash
modal serve model.py
```

4. Run the app
```bash
model serve server.py
```

5. Open your browser and visit the link displayed in the terminal!

### ENJOY! 🎉

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

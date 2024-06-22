import shlex, os
import subprocess
from pathlib import Path
import modal
from dotenv import load_dotenv
load_dotenv()

IMAGE_MODEL_DIR = "/root"


def download_llama():
    """
    Download the Llama model
    :return: None
    """
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)
    move_cache()

# ## Define container dependencies
#
# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.41.2",
    "accelerate",
    "streamlit", "bitsandbytes",
    "huggingface_hub", "hf_transfer",
    "langchain", "st-annotated-text",
    "python-dotenv", "scipy"
).env({
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SUNO_USE_SMALL_MODELS": "True"
})

app = modal.App(name="spellchecker-streamlit", image=image)

# ## Mounting the `app.py` script
#
# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

## Mounting the `app.py` script
streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# ## Spawning the Streamlit server
#
# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.


@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    timeout=60 * 20,
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)

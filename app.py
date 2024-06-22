import time
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    import streamlit as st
    from transformers import pipeline, VitsModel, AutoTokenizer, VitsConfig
    from annotated_text import annotated_text
    import requests
    st.set_page_config(
        page_title="Text Highlight and Correction App",
        page_icon="🔤",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    URL = os.getenv("LLM_URL")

    # Initialize the language correction pipeline

    @st.cache_resource
    def create_tts():

        return AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    @st.cache_resource
    def create_tts_model():
        config = VitsConfig.from_pretrained("facebook/mms-tts-eng", max_new_tokens=512)
        return VitsModel.from_pretrained("facebook/mms-tts-eng", config=config)

    @st.cache_resource
    def create_pipe():
        return pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

    @st.cache_data
    def llm_generate(text: str):
        response = requests.post(URL, json={"prompt": text})
        return response.json()


    if "pipe" not in st.session_state:
        st.session_state.pipe = create_pipe()

    if "tts" not in st.session_state:
        st.session_state.tts_processor = create_tts()
        st.session_state.tts_model = create_tts_model()

    def highlight_incorrect_words(text, corrected_text) -> [list, list]:
        """
        Highlight incorrect words in the text
        :param text:
        :param corrected_text:
        :return:
        """
        original_words = text.split()
        corrected_words = corrected_text.split()

        highlighted_text = []
        fixed_text = []
        for og, new in zip(original_words, corrected_words):
            if og != new:
                token1 = (f"**{og}** ", "incorrect", "#ff0000", '')
                token2 = (f"**{new}** ", "correct", "#07da63", '')
            else:
                token1 = og + ' '
                token2 = new + ' '
            highlighted_text.append(token1)
            fixed_text.append(token2)

        return highlighted_text, fixed_text

    st.title("Text Highlight and Correction App")
    user_input = st.text_input("Enter your text here:")
    if user_input:
        # Correct the text using langchain
        text = st.session_state.pipe(user_input, return_text=True, max_new_tokens=256)
        corrected_text = text[0]["generated_text"]

        # Highlight incorrect words
        wrong, right = highlight_incorrect_words(user_input, corrected_text)

        # Display the highlighted text
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            with st.container():
                col_1, col_2 = st.columns([1, 1])
                with col_1:
                    st.subheader("❎ This is the incorrect text")
                    annotated_text(*wrong)
                with col_2:
                    st.subheader("✅This is the Correct spelled text")
                    annotated_text(*right)
        with col2:
            with st.container():
                inputs = st.session_state.tts_processor(corrected_text, return_tensors="pt")
                co1, co2 = st.columns([1, 1])
                with co2:
                    st.subheader("🔊 Listen to the corrected text")
                    import torch
                    with torch.no_grad():
                        output = st.session_state.tts_model(**inputs).waveform
                    st.audio(output.float().numpy(), format="audio/wav", sample_rate=st.session_state.tts_model.config.sampling_rate)

                with co1:
                    st.subheader("✏ Spelling Correction Explained!")
                    output = llm_generate(user_input)

                    def stream_data():
                        for word in output.split():
                            yield word + ' '
                            time.sleep(0.05)

                    st.write_stream(stream_data)

        # Prevent the previous text from being displayed
        del user_input


if __name__ == "__main__":
    main()

import os
import openai
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import streamlit as st

STABILITY_KEY = st.secrets["STABILITY_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = st.secrets["MODEL"]
MODEL2 = st.secrets["MODEL2"]
openai.api_key = OPENAI_API_KEY

html_string = """<a href="https://platform.stability.ai">Generate Stability API Key using this link</a>"""

st.title('🎨 🔗 Image Generator App')

txt_input = st.text_input(r'Input Prompt:')
height = st.slider('Image Height', 256, 1024, step=8, value=512, key=1)
width = st.slider('Image Width', 256, 1024, step=8, value=512, key=2)

def generateImageViaStabilityai(prompt, stability_key=STABILITY_KEY):
    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
    stability_api = client.StabilityInference(
        key=stability_key, # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-xl-1024-v1-0", 
    )
    
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=prompt,
        # seed=42,      # If a seed is provided, the resulting generated image will be deterministic.
        steps=120,      # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=10.0, # Influences how strongly your generation is guided to match your prompt.
        width=width,    # Generation width, defaults to 512 if not included.
        height=height,  # Generation height, defaults to 512 if not included.
        style_preset="photographic", 
        samples=5, # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                #img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
                st.image(img, caption=f'Seed {artifact.seed}', use_column_width=True)

with st.form('submit_form', clear_on_submit=True):
    st.markdown(html_string, unsafe_allow_html=True)
    stability_api_key = st.text_input('Please, enter your Stability API Key', type='password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and stability_api_key.startswith('sk-'):
        with st.spinner('Generating ...'):
            try:
                generateImageViaStabilityai(prompt=txt_input, stability_key=stability_api_key)
            except Exception as e:
                st.info(e)
            del stability_api_key
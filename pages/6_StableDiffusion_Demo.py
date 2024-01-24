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

st.title('Image Generator App')

# Initialize session state if it doesn't exist
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = []
if 'selected_prompt' not in st.session_state:
    st.session_state['selected_prompt'] = ""
if 'edited_prompt' not in st.session_state:
    st.session_state['edited_prompt'] = ""


st.session_state['edited_prompt'] = st.text_input(r'Input Prompt:', value=st.session_state['selected_prompt'])


def generateImageViaStabilityai(prompt):
    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
    stability_api = client.StabilityInference(
        key=STABILITY_KEY, # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-xl-1024-v1-0", 
    )
    
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=prompt,
        seed=4253978046, # If a seed is provided, the resulting generated image will be deterministic.
                        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=1024, # Generation width, defaults to 512 if not included.
        height=1024, # Generation height, defaults to 512 if not included.
        style_preset="photographic", 
        samples=2, # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
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


# Button to generate the image
if st.button(r'generate image'):
    generateImageViaStabilityai(prompt=st.session_state['edited_prompt'])
    st.session_state['prompt_generated'] = False
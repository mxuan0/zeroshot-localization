import numpy as np
import streamlit as st
from PIL import Image
import torch, pdb
import clip

from utils.weighted_activation import weighted_activation
from utils.get_cam_weights import get_cam_weights
from utils.imageWrangle import heatmap, min_max_norm, torch_to_rgba
from utils.show import show_bbox_on_img, generate_bbox, scale_cam_image

st.set_page_config(layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache(show_spinner=True, allow_output_mutation=True)
def get_model():
    return clip.load("RN50", device=device, jit=False)

activations = {}
def get_features(name):
  def hook(model, input, output):
    activations[name] = output.detach()
  return hook

# OPTIONS:
st.sidebar.header('Options')
choose_model = st.sidebar.selectbox("select model", ['CLIP'], index=0)
alpha = st.sidebar.radio("select heatmap threshold", [0.55, 0.65, 0.75, 0.8], index=1)
layer = st.sidebar.selectbox("select saliency layer", ['layer4.2.relu3'], index=0)

st.header("Localization with image-text model")
st.write(
    "A quick experiment made for zero-shot object localization. ")
with st.expander('1. Upload Image', expanded=True):
    imageFile = st.file_uploader("Select a file:", type=[".jpg", ".png", ".jpeg"])

# st.write("### (2) Enter some desriptive texts.")
with st.expander('2. Write Descriptions', expanded=True):
    textarea = st.text_area("Descriptions about the object of interest", "an image of a ")
    # prefix = st.text_input("(optional) Prefix all descriptions with: ", "an image of")



to_find = st.button('Find it!')
if imageFile and to_find:
    st.markdown("<hr style='border:black solid;'>", unsafe_allow_html=True)
    image_raw = Image.open(imageFile)
    #pdb.set_trace()
    image_rgb = np.array(image_raw)
    model, preprocess = get_model()

    # preprocess image:
    image = preprocess(image_raw).unsqueeze(0).to(device)

    model.visual.layer4[2].relu3.register_forward_hook(get_features('act_map'))

    prompt = clip.tokenize([textarea]).to(device)
    with torch.no_grad():
        spatial_text_features = model.encode_text(prompt)
        spatial_text_features /= spatial_text_features.norm(dim=-1, keepdim=True)

        image_features = model.encode_image(image)
        activations_copy = activations["act_map"] 
    
    text_embedding = spatial_text_features.cpu().numpy()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # spatial_weights, base_scores = get_cam_weights(image, model, text_embedding, activations_copy.float(), device, BATCH_SIZE=32)
    # #pdb.set_trace()
    # saliency = weighted_activation(activations_copy[0].cpu().numpy(), spatial_weights[0], base_scores[0],
    #                         cap=True, softmax=True)
    
    saliency = np.arange(1,50).reshape((7, 7))
    scaled = scale_cam_image(saliency, target_size=(image_rgb.shape[0], image_rgb.shape[1]))
    cam_img = heatmap(torch.tensor(image_rgb), torch.tensor(scaled))
    
    props = generate_bbox(scaled, threshold=alpha, comp_size=image_rgb.shape[0]*image_rgb.shape[1]/1000)

    if not props:
        st.write('No instance found.')
    else:
        bbox_img = show_bbox_on_img(image_rgb, props)
        
        pdb.set_trace()

        st.write("### Localization Result and Heatmap for text input")
        st.image([Image.fromarray((torch_to_rgba(torch.tensor(bbox_img)).numpy() * 255.).astype(np.uint8)), 
                cam_img],
                caption=["localization result", "image heatmap"])  # , caption="Grad Cam for original embedding")

        # st.image(imageFile)


# @st.cache
def get_readme():
    with open('README.md') as f:
        return "\n".join([x.strip() for x in f.readlines()])


st.markdown("<hr style='border:black solid;'>", unsafe_allow_html=True)
# with st.beta_expander('Description', expanded=True):
#     st.markdown(get_readme(), unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Importing the required packages
import streamlit as st
import cv2
import numpy as np    
import tensorflow as tf
from models import get_unet,get_canet
import matplotlib.pyplot as plt

def main():
    #print(cv2.__version__)
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('About the Project','Evaluate the model','view source code')
        )
        
    #readme_text = st.markdown(get_file_content_as_string("README.md"))
    
    if selected_box == 'About the Project':
        st.sidebar.success('To try by yourself select "Evaluate the model".')
    if selected_box == 'Evaluate the model':
        #readme_text.empty()
        models()
    if selected_box=='view source code':
        #readme_text.empty()
        #st.code(get_file_content_as_string("app.py"))
        pass

def models():

    st.title('Image Segmentaion..')

    st.write('\n')
    
    #choice=st.sidebar.selectbox("Choose how to load image",["Use Existing Images","Browse Image"])

    uploaded_file = st.sidebar.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
      # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        gt = cv2.imdecode(file_bytes, 1)
        prediction_ui(gt)


# @st.cache(persist=True)
def get_models():
    unet=get_unet()
    #canet=tf.keras.models.load_model('CNET_15 - Copy.h5', compile=False)
    canet = get_canet()
    return unet,canet

def plot_output(output,gt):
    
    cv2.imwrite("out.jpg",output)

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.imshow(output)
    ax.axis('off')
    fig.savefig('temp.jpg',bbox_inches='tight', transparent = True, pad_inches = 0)   # save the figure to file
    plt.close(fig)
    output = cv2.imread('temp.jpg')
    output = cv2.resize(output, (gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_AREA) 
    st.image(output, clamp=True,channels="BGR")

def prediction_ui(gt):
    models_load_state=st.text('\n Loading models..')
    unet,canet=get_models()
    models_load_state.text('\n Models Loading..complete')
    st.header('Input Image')
    st.image(gt,channels="BGR")

    model = st.sidebar.radio("Choose a model to predict",('UNet', 'CANet'),0)
    submit = st.sidebar.button('Segment Now')

    progress_bar = st.progress(0)

    if submit and model=='UNet':
        output = predict(gt,unet,progress_bar).numpy()
        plot_output(output,gt)
        progress_bar.empty()

    if submit and model=='CANet':
        output = predict(gt,canet,progress_bar).numpy()
        plot_output(output,gt)    
        progress_bar.empty()

def predict(img,model,progress_bar):
    '''Funtion to iterate over each value in predicted 21 channel tensor and stack after assigning classs labels'''
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_AREA) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predicted  = model.predict(img[np.newaxis,:,:,:])

    progress_bar.progress(25)

    output=predicted[0]

    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         flag = True
    #         for k in range(output.shape[2]):
    #             if i==j and flag:
    #                 progress_bar.progress(25+k*2)
    #                 flag=False
    #             value=output[i][j][k]
    #             #print(value)
    #             if value>=0.2:  # Taking a threshold of 0.2 probability for class to exist
    #                 output[i][j][k]=1*k # Making the value to its corresponding class value based on its channel number
    #             else:
    #                 output[i][j][k]=0
    # progress_bar.progress(90)
    # output=tf.reduce_max(output,axis=-1) # Combining all the channel to a single channel
    # return output

    output[output>0.2] = 1
    for k in range(output.shape[2]):
        output[:,:,k] = output[:,:,k]*k
    return tf.reduce_max(output, axis=-1)

if __name__ == "__main__":
    main()





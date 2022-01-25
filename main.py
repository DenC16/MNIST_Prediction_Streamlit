from streamlit_drawable_canvas import st_canvas
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
import torchvision

st.write('# MNIST Digit Recognition')
st.write('## Using a CNN `PyTorch` model')

Network = torch.load('model_1.chk')


st.write('### Draw a digit in 0-9 in the box below')

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

realtime_update = st.sidebar.checkbox("Update in realtime", True)


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
 
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
)


if canvas_result.image_data is not None:

    input_numpy_array = np.array(canvas_result.image_data)
   
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')

    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200, 200)

    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)

    height, width = image.shape
    x, y, w, h = cv2.boundingRect(image)

    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10, ROI.shape[1]+10])
    width, height = mask.shape

    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2

    mask[y:y+h, x:x+w] = ROI

    output_image = Image.fromarray(mask)
    compressed_output_image = output_image.resize((22, 22))

    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    tensor_image = torch.nn.functional.pad(
        tensor_image, (3, 3, 3, 3), "constant", 0)
    
    convert_tensor = torchvision.transforms.Normalize(
        (0.1281), (0.3043))  
    tensor_image = convert_tensor(tensor_image)

    im = Image.fromarray(tensor_image.detach(
    ).cpu().numpy().reshape(28, 28), mode='L')
    im.save("processed_tensor.png", "PNG")
    
    plt.imsave('processed_tensor.png', tensor_image.detach(
    ).cpu().numpy().reshape(28, 28), cmap='gray')

    device = 'cpu'

    with torch.no_grad():
        output0 = Network(torch.unsqueeze(
            tensor_image, dim=0).to(device=device))


        certainty, output = torch.max(output0[0], 0)
        certainty = certainty.clone().cpu().item()
        output = output.clone().cpu().item()
        certainty1, output1 = torch.topk(output0[0], 3)
        certainty1 = certainty1.clone().cpu() 
        output1 = output1.clone().cpu()  

    st.write('### Prediction')
    st.write('### '+str(output))

    st.write('## Breakdown of the prediction process:')

    st.write('### Image being used as input')
    st.image(canvas_result.image_data)

    st.write('### Image as a grayscale Numpy array')
    st.write(input_image_gs_np)

    st.write('### Processing steps:')
    st.write('1. Find the bounding box of the digit blob and use that.')
    st.write('2. Convert it to size 22x22.')
    st.write('3. Pad the image with 3 pixels on all the sides to get a 28x28 image.')
    st.write('4. Normalize the image to have pixel values between 0 and 1.')
    st.write('5. Standardize the image using the mean and standard deviation of the MNIST training dataset.')

    st.write('### Processed image')
    st.image('processed_tensor.png')

    st.write('### Prediction')
    st.write(str(output))
    st.write('### Certainty')
    st.write(str(certainty1[0].item()*100) + '%')
    st.write('### Top 3 candidates')
    st.write(str(output1))
    st.write('### Certainties')
    st.write(str(certainty1*100))

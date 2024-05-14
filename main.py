import streamlit as st
import tensorflow as tf
import numpy as np



# Tensorflow Model Prediction
def model_prediction(test_image, selected_model):
    if selected_model == "CNN":
        model_path = 'trained_plant_disease_model.keras'
    elif selected_model == "VGG":
        model_path = 'fine_tuned_plant_disease_model_VGG.keras'
    else:
        st.error("Invalid model selection")
        

    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    top_indices = np.argsort(-predictions)[0][:3]
    top_confidences = predictions[0][top_indices]
    return top_indices, top_confidences

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
selected_model = st.sidebar.selectbox("Select Model", ["CNN", "VGG"])


# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "E:/Faculity Projects/Semester6_projects/Artificial Intelligence/CNN,VGG/Plant Health Inspection using CNN and VGG/home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍
        
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purposes.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)
        """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        top_indices, top_confidences = model_prediction(test_image,selected_model)
        class_names = [
            'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
            'Healthy Blueberry', 'Powdery Mildew on Cherry', 'Healthy Cherry',
            'Cercospora Leaf Spot on Corn', 'Common Rust on Corn', 'Northern Leaf Blight on Corn',
            'Healthy Corn', 'Black Rot on Grape', 'Esca (Black Measles) on Grape',
            'Leaf Blight (Isariopsis Leaf Spot) on Grape', 'Healthy Grape', 'Huanglongbing (Citrus Greening) on Orange',
            'Bacterial Spot on Peach', 'Healthy Peach', 'Bacterial Spot on Pepper', 'Healthy Pepper',
            'Early Blight on Potato', 'Late Blight on Potato', 'Healthy Potato', 'Healthy Raspberry',
            'Healthy Soybean', 'Powdery Mildew on Squash', 'Leaf Scorch on Strawberry',
            'Healthy Strawberry', 'Bacterial Spot on Tomato', 'Early Blight on Tomato',
            'Late Blight on Tomato', 'Leaf Mold on Tomato', 'Septoria Leaf Spot on Tomato',
            'Spider Mites Two-spotted Spider Mite on Tomato', 'Target Spot on Tomato',
            'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy Tomato'
        ]
        for i in range(len(top_indices)):
            st.success("Prediction {}: It's a {} with {:.2f}% confidence.".format(i + 1, class_names[top_indices[i]], top_confidences[i] * 100))
        # Add advice related to the predicted disease
        st.info("Advice: For optimal results, consult with a professional agronomist or refer to reliable agricultural resources for appropriate treatment and preventive measures.")

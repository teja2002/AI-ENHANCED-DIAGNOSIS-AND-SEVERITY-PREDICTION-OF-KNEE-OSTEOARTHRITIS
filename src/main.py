import os
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from datetime import datetime
import seaborn as sns

import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# Function to load feedback data from the Excel file
def load_feedback_data():
    file_path = './monitoring/feedback_responses.xlsx'
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist


# Scoring dictionary: each question has its respective options and scores
question_scores = {
    'Q1_Satisfaction': {
        "Very Satisfied": 2,
        "Satisfied": 1,
        "Neutral": 0,
        "Unsatisfied": -1,
        "Very Dissatisfied": -2
    },
    'Q2_Accuracy': {
        "Very Accurate": 2,
        "Accurate": 1,
        "Neutral": 0,
        "Inaccurate": -1,
        "Very Inaccurate": -2
    },
    'Q3_Ease_of_Use': {
        "Very Easy": 2,
        "Easy": 1,
        "Neutral": 0,
        "Difficult": -1,
        "Very Difficult": -2
    },
    'Q4_Meeting_Expectations': {
        "Exceeded Expectations": 2,
        "Met Expectations": 1,
        "Below Expectations": 0,
        "Did not meet expectations": -1
    },
    'Q5_Likelihood_of_Use': {
        "Very Likely": 2,
        "Likely": 1,
        "Neutral": 0,
        "Unlikely": -1,
        "Very Unlikely": -2
    }
}

# Function to calculate the total score based on feedback data
def get_question_scores(feedback_data):
    total_score = 0
    for question, response in feedback_data.items():
        if question in question_scores:
            score = question_scores[question].get(response, 0)
            total_score += score
    return total_score



def save_feedback(feedback_data):
    # Calculate the total score
    # total_score = get_question_scores(feedback_data)
    total_score = feedback_data["Total_Score"]
    
    # Add the total score to feedback data
    # feedback_data["Total_Score"] = total_score

    # Define the new path to the feedback file
    feedback_file = './monitoring/feedback_responses.xlsx'
    
    # Add timestamp to feedback_data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format the timestamp as 'YYYY-MM-DD HH:MM:SS'
    
    # If feedback_data is a dictionary, wrap it in a list and add timestamp
    if isinstance(feedback_data, dict):
        feedback_data = [feedback_data]
    
    # Add timestamp to each entry in the feedback data
    for entry in feedback_data:
        entry['Timestamp'] = timestamp
    
    # Create a DataFrame from feedback_data
    df = pd.DataFrame(feedback_data)
    # Reorder columns to place Timestamp first
    cols = ["Timestamp"] + [col for col in df.columns if col != "Timestamp"]
    df = df[cols]

    # Check if the feedback file exists
    if not os.path.exists(feedback_file):
        # If it doesn't exist, create a new file and save the responses with column names
        df.to_excel(feedback_file, index=False, header=True)
    else:
        # If the file exists, append the responses correctly without headers
        with pd.ExcelWriter(feedback_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            # Load existing data to determine the row position for appending
            existing_df = pd.read_excel(feedback_file, engine='openpyxl')
            startrow = len(existing_df)  # Get the row count of existing data to append at the end
            df.to_excel(writer, index=False, header=False, startrow=startrow+1)




def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


# Page Configuration
st.set_page_config(page_title="AI Diagnosis", page_icon="🩺", layout="wide")
# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["Diagnosis", "Feedback", "Admin panel"])

if page == "Diagnosis":

    # st.set_page_config(page_title="AI Enhanced diagnosis & Severity prediction of knee OA")
    # Page Header
    st.title("AI-Enhanced Diagnosis and Severity Prediction of Knee OA")
    st.markdown("""
    Welcome to the **AI-Enhanced Diagnosis Tool for Knee Osteoarthritis**!  
    This application uses state-of-the-art **Deep Learning** techniques, like Grad-CAM, to diagnose and classify the severity of **Knee Osteoarthritis (OA)** from X-ray images.

    ### How it Works
    1. **Upload an X-ray Image**: Use the sidebar to upload an X-ray image of the knee.
    2. **Analyze Results**: The AI will predict the severity grade (Healthy, Doubtful, Minimal, Moderate, Severe) and display the probability.
    3. **Explainability**: Visualize the **heatmap** showing the regions that contributed most to the prediction.
    4. **Get Insights**: A bar chart shows confidence levels for all possible grades.

    """)


    class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
    model = tf.keras.models.load_model("./src/models/model_ResNet50_ft.hdf5")
    target_size = (224, 224)

    # Grad-CAM model setup
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[grad_model.get_layer("global_average_pooling2d_1").input, grad_model.output]
    )

    y_pred = 0
    grade = None
    probability = 0

    # Sidebar
    with st.sidebar:
        st.subheader("EGN 6216 - AI Systems")
        # st.subheader(":arrow_up: Upload image")
        uploaded_file = st.file_uploader("Choose x-ray image")
        if uploaded_file is not None:
            app_start_time = time.time()
            st.subheader("Uploaded Image:-")
            st.image(uploaded_file, use_column_width=False, width = 200)

            img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img_aux = img.copy()

            if st.button("Diagnose Knee OA"):
                img_array = np.expand_dims(img_aux, axis=0)
                img_array = np.float32(img_array)
                img_array = tf.keras.applications.xception.preprocess_input(img_array)

                with st.spinner("Wait for it..."):
                    model_start_time = time.time()
                    y_pred = model.predict(img_array)
                    model_end_time = time.time()
                    model_latency = model_end_time - model_start_time
                
                y_pred = 100 * y_pred[0]
                probability = np.amax(y_pred)
                number = np.where(y_pred == np.amax(y_pred))
                grade = str(class_names[np.amax(number)])

                # st.subheader(":white_check_mark: Severity Grade:")    
                # st.metric(label="Severity Grade:", value=f"{grade} - {probability:.2f}%")
                

    # Body
    # st.header("AI ENHANCED DIAGNOSIS AND SEVERITY PREDICTION OF KNEE OSTEOARTHRITIS")

    col1, col2 = st.columns([1, 1])  # Left column for uploaded image and Predict button



    if uploaded_file!=None and probability!=0 and grade!=None:
        with col1:
            st.subheader("Analysis Report:")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(class_names, y_pred, height=0.55, align="center")
            for i, (c, p) in enumerate(zip(class_names, y_pred)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")
            ax.grid(axis="x")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            fig.tight_layout()
            st.pyplot(fig)
            app_end_time = time.time()
            app_latency = app_end_time - app_start_time

            # st.subheader("Diagnosis:")
            st.write(f"**Grade:** {grade}")  # Display grade
            st.write(f"**Probability:** {probability:.2f}%")  # Display probability
            st.write(f"**Model Latency:** {model_latency:.2f} second(s)")  # Model latency
            


            # print("")

        if probability!=0 and grade!=None:
            with col2:
                st.subheader("GradCam Explainability")
                heatmap = make_gradcam_heatmap(grad_model, img_array)
                image = save_and_display_gradcam(img, heatmap)
                st.image(image, use_column_width=False, width = 250)
                st.write(f"**Application Latency:** {app_latency:.2f} second(s)")  # Application latency

            # Feedback link below results
            # st.markdown(
            #     """
            #     <a href="https://forms.gle/MjqbmZNKqL34UuNw5" target="_blank" style="font-size: 16px; background-color: #008CBA; color: white; padding: 10px; border-radius: 5px; text-decoration: none;">Give Feedback</a>
            #     """, unsafe_allow_html=True
            # )
            

elif page == "Feedback":
    # Sidebar Information for Feedback
    st.sidebar.title("Feedback Instructions")
    st.sidebar.markdown("""
        **We appreciate your feedback!**
        
        Please provide your feedback regarding the predictions made by the tool. Your responses will help us improve the tool and your experience.
        
        After you submit the form, you will be able to see how others have rated the tool (admin panel).
        
        Thank you for your time and valuable feedback!
    """)

    st.title("Feedback Form")
    st.markdown("We value your feedback! Please fill out the form below.")

    # Collecting feedback for 5 questions
    feedback_q1 = st.radio("1. Overall, how satisfied are you with the predictions made by the tool?", 
                            ["Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very Dissatisfied"])
    feedback_q2 = st.radio("2. How accurate did you find the severity prediction of the knee osteoarthritis grade?", 
                            ["Very Accurate", "Accurate", "Neutral", "Inaccurate", "Very Inaccurate"])
    feedback_q3 = st.radio("3. How easy was it to upload your image and receive the results?", 
                            ["Very Easy", "Easy", "Neutral", "Difficult", "Very Difficult"])
    feedback_q4 = st.radio("4. Was the tool able to meet your expectations for diagnosing knee osteoarthritis severity?", 
                            ["Exceeded Expectations", "Met Expectations", "Below Expectations", "Did not meet expectations"])
    feedback_q5 = st.radio("5. How likely are you to use this tool again in the future?", 
                            ["Very Likely", "Likely", "Neutral", "Unlikely", "Very Unlikely"])

    feedback_name = st.text_input("Name")
    feedback_email = st.text_input("Email")

    if st.button("Submit Feedback"):
        # Map responses to scores
        feedback_scores = {
            "Q1_Satisfaction": question_scores['Q1_Satisfaction'][feedback_q1],
            "Q2_Accuracy": question_scores['Q2_Accuracy'][feedback_q2],
            "Q3_Ease_of_Use": question_scores['Q3_Ease_of_Use'][feedback_q3],
            "Q4_Meeting_Expectations": question_scores['Q4_Meeting_Expectations'][feedback_q4],
            "Q5_Likelihood_of_Use": question_scores['Q5_Likelihood_of_Use'][feedback_q5]
        }
        total_score = sum(feedback_scores.values())
        # print(total_score)
        # Store both options and scores
        feedback_data = {
            "Name": feedback_name,
            "Email": feedback_email,
            "Q1_Satisfaction_Option": feedback_q1,
            "Q1_Satisfaction_Score": feedback_scores['Q1_Satisfaction'],
            "Q2_Accuracy_Option": feedback_q2,
            "Q2_Accuracy_Score": feedback_scores['Q2_Accuracy'],
            "Q3_Ease_of_Use_Option": feedback_q3,
            "Q3_Ease_of_Use_Score": feedback_scores['Q3_Ease_of_Use'],
            "Q4_Meeting_Expectations_Option": feedback_q4,
            "Q4_Meeting_Expectations_Score": feedback_scores['Q4_Meeting_Expectations'],
            "Q5_Likelihood_of_Use_Option": feedback_q5,
            "Q5_Likelihood_of_Use_Score": feedback_scores['Q5_Likelihood_of_Use'],
            "Total_Score": total_score
        }
        save_feedback(feedback_data)
        st.success("Thank you for your feedback!")

elif page == "Admin panel":
    st.sidebar.title("Admin Metrics Overview")
    st.sidebar.markdown("""
        **Feedback Score Analysis**
        
        This section provides insights into user feedback collected through the tool. 
        
        - **Score Range**: Feedback scores range from **-10 (very negative)** to **+10 (very positive)**.
        - **Metric Details**: Each feedback score is calculated by assigning values to responses for each question, then summing them up.
        - **Use Case**: Use this data to evaluate user satisfaction and identify areas for improvement.
        
        **Tips**:
        - Look for trends in the data over time.
        - Focus on scores below 0 to address user concerns.
    """)

    st.title("Admin panel: User feedback metrics")
    st.markdown("Here you can track the feedback scores from users.")

    # Load feedback data
    df = load_feedback_data()

    # Check if there is feedback data available
    if not df.empty:
        # Plot the feedback score
        st.subheader("Feedback Score Distribution")

        # Create a seaborn plot
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(data=df, x='Name', y='Total_Score', palette='viridis')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=8, color='black', 
                        xytext=(0,8 if p.get_height()>= 0 else -8),
                        textcoords='offset points')

        plt.ylim(-10, 10)
        plt.xlabel('User Name', fontsize = 8)
        plt.ylabel('Total Feedback Score', fontsize = 8)
        # plt.title('Feedback Scores for Each User')
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)


        # Plotting user-wise scores for each question
        st.subheader("User-wise Scores for Each Question")

        # Extract the user-wise scores for each question (Q1 to Q5)
        question_scores = df[['Name', 'Q1_Satisfaction_Score', 'Q2_Accuracy_Score', 'Q3_Ease_of_Use_Score', 'Q4_Meeting_Expectations_Score', 'Q5_Likelihood_of_Use_Score']]

        # Set up the figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set the x-axis positions for each user
        x = range(len(df))

        # Define colors for each question
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Plot each question's scores as bars for each user
        ax.bar(x, df['Q1_Satisfaction_Score'], width=0.15, label='Q1: Satisfaction', color=colors[0], align='center')
        ax.bar([i + 0.15 for i in x], df['Q2_Accuracy_Score'], width=0.15, label='Q2: Accuracy', color=colors[1], align='center')
        ax.bar([i + 0.30 for i in x], df['Q3_Ease_of_Use_Score'], width=0.15, label='Q3: Ease of Use', color=colors[2], align='center')
        ax.bar([i + 0.45 for i in x], df['Q4_Meeting_Expectations_Score'], width=0.15, label='Q4: Expectations', color=colors[3], align='center')
        ax.bar([i + 0.60 for i in x], df['Q5_Likelihood_of_Use_Score'], width=0.15, label='Q5: Likelihood of Use', color=colors[4], align='center')

        # Set labels and title
        ax.set_xlabel('Users', fontsize=12)
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title('Scores for Each User Across Questions', fontsize=14)
        ax.set_xticks([i + 0.3 for i in x])  # Position the x-axis ticks in the middle of the bars
        ax.set_xticklabels(df['Name'], rotation=45, ha='right', fontsize=10)

        # Add a legend
        ax.legend(bbox_to_anchor = (1.10, 1), loc = 'upper left', fontsize=10)

        # Tight layout to prevent overlap
        plt.tight_layout()
        st.pyplot(fig)


        # Total scores for each question (sum of all users' scores)
        st.subheader("Total Score for Each Question")

        # Sum up scores for each question
        total_scores = {
            'Q1_Satisfaction_Score': df['Q1_Satisfaction_Score'].sum(),
            'Q2_Accuracy_Score': df['Q2_Accuracy_Score'].sum(),
            'Q3_Ease_of_Use_Score': df['Q3_Ease_of_Use_Score'].sum(),
            'Q4_Meeting_Expectations_Score': df['Q4_Meeting_Expectations_Score'].sum(),
            'Q5_Likelihood_of_Use_Score': df['Q5_Likelihood_of_Use_Score'].sum()
        }

        # Create a bar chart to display total scores for each question
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=list(total_scores.keys()), y=list(total_scores.values()), palette='viridis')

        # Add the total score above each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=10, color='black', 
                        xytext=(0, 8 if p.get_height() >= 0 else -8),
                        textcoords='offset points')

        plt.xlabel('Questions', fontsize=12)
        plt.ylabel('Total Scores', fontsize=12)
        plt.xticks(rotation=45, fontsize=8)
        plt.title('Total Scores for Each Question', fontsize=14)
        plt.tight_layout()
        st.pyplot(plt)


    else:
        st.warning("No feedback data available.")
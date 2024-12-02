# AI-ENHANCED DIAGNOSIS AND SEVERITY PREDICTION OF  KNEE OSTEOARTHRITIS
## 1. Project Title and Overview

AI-ENHANCED DIAGNOSIS AND SEVERITY PREDICTION OF  KNEE OSTEOARTHRITIS.



## 2. Repository Contents

`src/:` It contains 
      **Combined code testings**(where we both tested together), **kolla.teja** (Files trained and tested by teja), **manukonda.g**(Files trained and tested by gopal),  **dataset**(dataset), **models** (folder is currently empty because the file exceeds 100 mb to push so we uploaded to drive), **utils**(final code notebook) and it also consists of **environment file, requirements file and streamlit file**(main.py)

`deployment/:` It consists dockerfile and requirements file for docker.

`monitoring/:` Feedback_responses.xslx for saving responses

`documentation/:` Templates and reports documenting and Readme file

`videos/:` Demo screencast showing the system in action named system_demo in both mp4 and webm extensions.

## 3. System Entry Point
Main script: src/main.py

This file consists of code to run the model using streamlit to run locally using the below command.


```bash
streamlit run ./src/main.py
```

Saved model took more than 250 mb space, so we uploaded it into google drive and gave everyone access.

Link: ```[https://drive.google.com/file/d/1GbAmU9CN4laQErTA2aGJxAK18e4CpACX/view?usp=sharing](https://drive.google.com/file/d/1GbAmU9CN4laQErTA2aGJxAK18e4CpACX/view?usp=sharing)```

## 4. Video Demonstration

In the video demonstration we clearly stated how the model and system works after deployment, the file is named as system_demo and for easy access we have given you the two extensions such as mp4 and webm.

## 5. Deployment Strategy

Initially We deployed our model to **streamlit** (Streamlit is an open-source Python library for quickly building and sharing interactive, data-driven web applications with minimal code.) to run it locally and then we pushed that to **docker** (Docker is an open-source platform that allows developers to automate the deployment of applications in lightweight, portable containers that include everything needed to run them) and created images and containers to run them. We are running it on port 8501.

To create the docker image and run the container below is the code.

```bash
# Build the Docker image
docker build -t streamlit-ais-app .

# Run the container
docker run -p 8501:8501 streamlit-ais-app
```



## 6. Monitoring and Metrics
We have incorporated an metrics inside our deployed model and given a feedback form for the user, with that we can calculate the metrics real time by storing the data in the feedback_responses.xlsx file and by using openpyxl to load the data real time. Latency is calculated all the time for both prediction and gradcam.

## 7. Project Documentation
AI System Project Proposal: documentation/AI System project proposal template

Project Report : documentation/Project report

## 8. Version Control and Team Collaboration

Pushed all the contents to the main branch and you can find up to data versions over there. 

`Teja Kolla:` 
1. Developed the core deep learning models, including ResNet50 for severity prediction.

2. Experimented with hyperparameter tuning to optimize model performance.

3. Integrated explainability tools (e.g., Grad-CAM) to visualize and interpret model predictions.
4. Designed and implemented the Streamlit-based application for user-friendly interactions.

5. Incorporated functionalities for users to upload X-ray images and receive instant feedback on severity.

6. Ensured real-time deployment and efficient model inference within the app.

7. Focused on testing the model’s performance in real-time conditions.

8. Finalized deployment strategies for Streamlit.

9. Designed CI/CD pipelines for updates and smooth application rollouts.

10. Maintained a README file to guide collaborators through the repository structure.

`Gopalakrishna Manukonda:` 

1. Conducted extensive preprocessing of knee X-ray images, including resizing, normalization, and augmentation.

2. Analyzed dataset quality and handled data imbalances through techniques like oversampling and class weighting.

3. Prepared train-test-validation splits and ensured dataset consistency.

4. Drafted sections on data management, risk analysis, and validation outcomes.
5. Designed the project structure for clarity and scalability
6. Performed unit testing for the Streamlit application and end-to-end system validation.
7. Interacted with stakeholders to gather requirements and feedback.

8. Addressed stakeholder concerns related to ethical implications, model biases, and practical deployment.
9. Established monitoring metrics and logging for app performance.
10. Focused on understanding data collection methodologies and ethical data handling.


## 9. If Not Applicable
We haven't used `prometheus` and `grafana` instead we directly incorporated metrics into the system as separate window named admin panel where you can find all the metrics.

## 10. Issues

If you face any issues creating docker image then it maybe because of several reasons and one of them is file structure is not detected, in that case place the files : **main.py, dockerfile, requirements.txt, model_ResNet50_ft.hdf5, feedback_resposnes.xlsx**


The docker file should look like this. Below is our dockerfile code

```python
# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY main.py /app
COPY model_ResNet50_ft.hdf5 /app
COPY feedback_responses.xlsx /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
## 11. Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## 12. License

[MIT](https://choosealicense.com/licenses/mit/)

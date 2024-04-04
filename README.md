# Early Prediction of Parkinsons Disease using Simple Machine Learning

## Introduction:
Parkinson's disease (PD), a widely observed neurological condition affecting muscle coordination, presents with symptoms such as tremors, muscle rigidity, and bradykinesia. The fundamental cause is the degeneration of neurons, resulting in reduced dopamine levels that disrupt synaptic communication, leading to compromised motor functions. Although the progression of the disease varies from person to person, balance issues and tremors consistently emerge as primary consequences of dopaminergic neuron loss. Despite its global prevalence, PD remains without a cure, emphasizing the importance of early detection and personalized interventions to manage its progression.
PD unfolds in five progressive stages, with up to 90% of patients with Parkinson's (PWP) displaying vocal cord impairments as an initial symptom in stage 0. Vocal dysfunction, easily quantifiable and falling within the purview of telemedicine, allows for remote monitoring, enabling patients to undergo audio tests at home. Vocal modulation symptoms, including dysphonia and dysarthria, are evaluated through sustained phonation tests or running speech assessments, providing realistic indicators of impairment. The potential for early detection in stage 0 creates a crucial window for intervention. The global impact of PD and the challenges associated with its diagnosis, often costly and lacking in accuracy, underscore the pressing need for cost-effective and precise diagnostic techniques. While previous models incorporated sophisticated technologies like Deep Learning and Neural Networks, requiring significant computational resources, this project adopts a machine learning approach with a focus on deploying simpler models. This strategy aims to improve affordability, widen accessibility, and contribute to effective PD identification in various applications. 

## Methodology
Following is the block diagram of methodology implied in this study-

![WhatsApp Image 2024-02-08 at 22 34 15_82bd05e1](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/41adf7eb-172c-48d4-aefe-61f25085b8ff)
### Data Collection

The dataset utilized for this study, curated by Little, McSharry, Hunter, Spielman, and Ramig, consists of various biomedical voice measurements from 31 people, with 23 patients having PD and the rest being healthy [1]. The dataset is in ASCII CSV format and includes the following columns:

#### Table 1: Dataset's Attribute Information

| Voice Features        | Description                                                |
|-----------------------|------------------------------------------------------------|
| MDVP:Jitter(%)        | Measure of fundamental frequency variation in percentage.  |
| MDVP:Jitter(Abs)      | Absolute measure of fundamental frequency variation.       |
| MDVP:PPQ              | Measure related to fundamental frequency variation.        |
| MDVP:Fo(Hz)           | Fundamental frequency of the voice.                        |
| MDVP:Flo(Hz)          | Lowest fundamental frequency.                             |
| MDVP:Fhi(Hz)          | Highest fundamental frequency.                            |
| MDVP:Shimmer          | Amplitude variation in the voice signal.                  |
| MDVP:Shimmer(dB)      | Amplitude variation in decibels.                          |
| Shimmer APQ3          | Amplitude variation measured in three-point quartile.     |
| Shimmer APQ5          | Amplitude variation measured in five-point quartile.      |
| MDVP:APQ              | Measure related to amplitude variation.                   |
| Shimmer:DDA           | Amplitude variation calculated using the DDA algorithm.   |
| NHR                   | Ratio of noise and tonal components in the voice.         |
| HNR                   | Ratio of tonal components in the voice.                    |
| RPDE                  | Measure of nonlinear dynamical complexity.                |
| DFA                   | Exponent of signal fractal scaling.                       |
| spread1, spread2, PPE | Nonlinear measures of fundamental frequency variation.     |

Each row in the dataset represents one of the 5,875 records of these individuals. The dataset is in ASCII CSV format, with each column representing an event related to a record. Each patient has approximately 200 records, with the patient number listed on the first line.

# Data Preprocessing

The dataset was imported and read using the Pandas library. Duplicate rows were checked for due to the binary nature of the dataset and the intended utilization of classifiers such as Decision Tree, for which the existence of duplicated rows is not recommended. No duplicate rows were identified. Following this, scrutiny for any instances of missing data was undertaken. Various interpolating methods offered by Pandas can be employed to address missing data; however, no instances of missing values were detected in our study.

#### Table 2: Type and Number of Data Values for Each Input Variable

### Table 2: Type and Number of Data Values for Each Input Variable

| # | Column           | Non-Null Count | Dtype    |
|---|------------------|----------------|----------|
| 0 | name             | 195 non-null   | object   |
| 1 | MDVP:Fo(Hz)      | 195 non-null   | float64  |
| 2 | MDVP:Fhi(Hz)     | 195 non-null   | float64  |
| 3 | MDVP:Flo(Hz)     | 195 non-null   | float64  |
| 4 | MDVP:Jitter(%)   | 195 non-null   | float64  |
| 5 | MDVP:Jitter(Abs)| 195 non-null   | float64  |
| 6 | MDVP:RAP         | 195 non-null   | float64  |
| 7 | MDVP:PPQ         | 195 non-null   | float64  |
| 8 | Jitter:DDP       | 195 non-null   | float64  |
| 9 | MDVP:Shimmer     | 195 non-null   | float64  |
|10 | MDVP:Shimmer(dB) | 195 non-null   | float64  |
|11 | Shimmer:APQ3     | 195 non-null   | float64  |
|12 | Shimmer:APQ5     | 195 non-null   | float64  |
|13 | MDVP:APQ         | 195 non-null   | float64  |
|14 | Shimmer:DDA      | 195 non-null   | float64  |
|15 | NHR              | 195 non-null   | float64  |
|16 | HNR              | 195 non-null   | float64  |
|17 | status           | 195 non-null   | int64    |
|18 | RPDE             | 195 non-null   | float64  |
|19 | DFA              | 195 non-null   | float64  |
|20 | spread1          | 195 non-null   | float64  |
|21 | spread2          | 195 non-null   | float64  |
|22 | D2               | 195 non-null   | float64  |
|23 | PPE              | 195 non-null   | float64  |

Table 3: List of Data info, ‘non-null’ indicates the absence of missing data within the dataset.

### 3.3 EDA (Exploratory Data Analysis)
Upon executing Data Preprocessing, it becomes imperative to assess the class distribution. An examination of the distribution uncovered an imbalanced class structure, with 75.4% of the dataset representing patients diagnosed with Parkinson's disease, while the remaining 24.6% corresponds to healthy individuals.

![WhatsApp Image 2024-02-08 at 22 34 15_9f5d3442](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/2a8beb0b-46ad-4342-9180-b89d034b5a6a)
Figure 2: Pie Plot of Class Distribution

To address the identified class imbalance and further refine model accuracy upon integration, an in-depth analysis was conducted leveraging histograms, box-plots and a correlation heatmap. By this, the presence of outliers could be pinpointed and insights into the skewness as well as the correlation nature associated with the input parameters could be gained.

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/6b764a7a-e432-4c7d-88b9-522f8ce26369)
Figure 3: Histogram and Box Plot of 4 randomly selected input parameters

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/53e5966d-6f5e-42b3-8298-2d0745b25108)
Figure 4: Correlation Heatmap

It was observed from Figure 3 that some input parameters consisted outliers and most of them showed asymmetric skewness. Also, from Figure 4 it can be observed that few of the input parameters were highly correlated with each other. These factors can affect the model performance drastically since it would be a challenge to accurately predict the minority values. Moreover, highly correlated input parameters provide insignificant information and creates difficulty for models to generalize. Hence only some significant input parameters must be selected to prevent this issue. We then proceeded to check for statistical significance of each input parameter.

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/bbc3b9c9-c0ea-457d-bc47-b96af5af724b)
Figure 5: Box Plot with measure of Variance Ratio, T-Test and Welch T-Test for ‘RPDE’ input parameter

Ratio of Variance, T-Test and Welch T-Test were calculated for each input parameter. Since a few of input parameters had irregular distributions, we calculated T-Test and Welch T-Test ratios together for each input parameter to reduce lines of code. It was observed that all input parameters were statistically significant as p-values for either t-test or welch t-test for all features were below 0.5.

### 3.4 Initial Model Building with imbalanced class
Given the statistical significance observed across all input parameters, no modifications in the original features were made to the dataset. Before we could proceed, we created an instance of the original dataset which had all columns except for ‘Name’ and ‘Status’ since they weren’t required for supervised learning. Subsequently, the dataset was partitioned into Test (20%) and Train data (80%). We proceeded to apply nine distinct classifiers, namely RF, XGB, KNN, ADB, DT, RC, LR, SVM, and NB. To systematically evaluate the performance of each classifier, a dedicated function was formulated. This function calculated essential performance metrics, including AUC, Accuracy, Precision, Recall, and F1 Score, presenting the results in a tabular format for comprehensive analysis. For this study, we consistently set the random state to 9 for each model, with the exception of Naïve Bayes and KNN.

### 3.5 Data Standardization
The standardization process, a fundamental step in data preprocessing, entails converting data into a uniform format. In this case, we employed the Standard Scaler, applying the formula - 

z =(x-μ)/σ  

where x represents an individual data point (sample), μ denotes the mean of the dataset, σ signifies the standard deviation of the dataset and z represents the standardized value. The Standard Scaler module from the Scikit Learn library was imported and applied to standardize both the Test and Train datasets. This standardization was deemed necessary due to the dataset encompassing both negative and positive scores across varying ranges, ultimately enhancing the performance of the classifiers.

### 3.6 Class Balancing
Following the standardization of data, the subsequent step involved addressing class imbalance. Given the inherent imbalance between patients diagnosed with PD and healthy patients within our dataset, it was imperative to balance the classes. Class imbalance can significantly impede model performance, and to mitigate this, we employed the Synthetic Minority Oversampling Technique (SMOTE) module from imblearn. SMOTE involves oversampling the minority class, in this case, healthy patients, to constitute 50% of the dataset. This strategic oversampling approach effectively balanced the data distribution, thereby enhancing the overall performance of the model.

### 3.7 Fitting Classifiers over balanced data
Following the class balancing, we fitted the nine distinct classifiers over this balanced data and calculated performance metrics for each classifier using the same function as the one used in 3.4.

### 3.8 Feature Engineering
After standardizing the data and addressing class imbalance, the subsequent step in our study was feature engineering. During the Exploratory Data Analysis, we generated a correlation heatmap (refer to Figure 4) to visualize the correlations among various input parameters. We observed notable positive and negative correlations between several input parameters. Such correlations can detrimentally impact model performance. Since all parameters were found to be statistically significant, we created a function to plot the importance of each feature. The function calculated the cumulative sum of correlation factors for the sorted data and features with a cumulative sum exceeding 0.8 were then selected and color-coded in green, while the remaining features were color-coded in red in the bar plot. The function also displayed the feature-importance value next to each feature. This method allowed us to identify and prioritize features that collectively contributed significantly to the dataset. This function was utilized on another instance of the original un-modified dataset after dropping ‘Name’ and ‘status’ column.

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/2c330124-182d-4179-9d3d-9594129fdedc)
Figure 6: Bar Plot of Importance of each Feature

We found that there were 12 important features, namely PPE, spread1, spread2, MDVP:APQ, MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Shimmer, Shimmer:APQ5, Shimmer:DDA, MDVP:Jitter(Abs), D2, MDVP:Flo(Hz). 

### 3.9 Final model selection and loading:
After dropping all other variables and retaining only the important ones, the subsequent step in our study involved oversampling the new unmodified instance of original data using SMOTE. This was done to address potential imbalances introduced during the feature selection process. Following oversampling, the dataset was split into train and test sets. To maintain the integrity of the data and prevent the model from being biased toward certain characteristics, standardization was then performed. This post-split standardization ensures that both the training and testing sets are scaled consistently, preserving the relative relationships between features. We then proceeded with fitting the models and using the previous performance metric calculating function to create a tabular representation of each classifier’s performance arranged in a descending order of Accuracy. From this table, we observed the model having the highest accuracy and selected it as the Final model.

### 3.10 User Input Form
Following the selection of the final model, we developed an interface to facilitate user input and obtain predicted results. This interface was created through a web application developed in Python using Flask framework and hosted on PythonAnywhere. Users could input values via a form on a dedicated website. Upon clicking the "Submit" button, the application processed the input data and generated a new webpage (refer to Figure 8 and Figure 9), presenting the prediction outcome.

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/e2568813-7910-4481-96cd-70eee3c400ec)
Figure 7: Input form as a Web Application

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/c6d7faab-c0eb-4462-b3cf-ee247099538e)
Figure 8: Result Displayed if user is predicted to be suffering from PD

![image](https://github.com/Bhagy3sh/Simple_Parkinsons_Disease_Prediction/assets/141747782/c3181061-f426-4bc3-8dff-b74854e462a5)
Figure 9: Result Displayed of user is predicted to be healthy

The prediction process was executed by utilizing a pre-trained model file, saved using the Pickle library. Within the main application file, we loaded this model, and the calculated predicted result was then displayed on the web page. This interactive and user-friendly approach allowed individuals to receive predictions regarding the likelihood of having Parkinson's disease based on their provided input.

## RESULTS
After following the stepwise proposed methodology and the system architecture, the key findings are as follows –
Table 4: Model Performance for Imbalanced Data













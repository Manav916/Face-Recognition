# Face-Recognition


This project is aimed at building a comprehensive face recognition system with emotion detection capabilities. The project encompasses multiple phases, including machine learning, deep neural networks, and web application development using Django.

<table><tr>
<td> <img src= "https://github.com/Manav916/Face-Recognition/assets/77217074/b5e66f6b-1056-40dd-83a1-e49ebf82fc5b" width="100%" align="center"  hspace="5%" vspace="5%"/> </td>
<td> <img src= "https://github.com/Manav916/Face-Recognition/assets/77217074/1586fbb7-7e59-4ebd-a881-c512af2b7d2e" width="100%" align="center"  hspace="5%" vspace="5%"/> </td>
</tr></table>


## Project Overview

In this project, we have implemented the following phases:

### Phase 1: Machine Learning - Face Identification

In this phase, we perform face identification using machine learning techniques. Here's an overview of the steps involved:

1. **Data Preprocessing:** We preprocess the images, ensuring they are suitable for training.

2. **Feature Extraction:** Deep neural networks are used to extract features from facial images.

3. **Modelling:** We train various machine learning models, such as logistic regression, support vector machines, and random forest, using the features extracted from celebrity images. 

4. **Ensemble Learning:** A voting classifier is employed to combine the predictions of individual models.

5. **Hyperparameter Tuning:** We fine-tune the hyperparameters to optimize the final model.

### Phase 2: Machine Learning - Facial Emotion Recognition

In this phase, we focus on emotion recognition using convolutional neural networks (CNNs). The process is as follows:

1. **Data Preprocessing:** We preprocess the images, ensuring they are suitable for training.

2. **Modelling:** We train a convolutional neural network to recognize emotions from facial expressions.

### Phase 3: Django Web App Development

In the final phase, we create a web application using Django. The development process includes the following steps:

1. **Web Frontend:** We render the frontend using HTML, CSS, and Bootstrap for an appealing user interface.

2. **Backend:** The backend is powered by Python and the Django framework, following the MVT (Models, Views, Templates) architecture.

3. **Database:** We design and utilize an SQLite database for data storage.

4. **Integration:** Both the face recognition and emotion detection models are integrated into the web app to provide face analysis.

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository to your local machine.
   
2. Navigate to the project directory.

3. Install the necessary dependencies using the following command:

   ```
   pip install -r requirements.txt
   ```

4. Get SECRET_KEY:

   ```
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```

5. Run the Django development server:

   ```
   python manage.py runserver
   ```

6. Access the web application by opening a web browser and entering the following URL:

   ```
   http://localhost:8000/
   ```


## Contributions

Contributions to this project are welcome! If you have any improvements or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the open-source community for their valuable contributions and the dataset providers for making their data available for research and development.


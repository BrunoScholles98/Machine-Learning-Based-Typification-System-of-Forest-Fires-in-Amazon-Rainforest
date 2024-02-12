## Contents

1. [Description](#desc)
2. [Project Detailing](#det)
3. [Contact](#contact)
4. [Thanks](#thanks)

# Machine Learning-based Typification System of Forest Fires in Amazon Rainforest

<a name="desc"></a>
## 1. Description

The focus of this repository is dedicated to a research project aimed at utilizing machine learning and deep learning techniques for detecting fire events in the Legal Amazon region. This initiative was carried out in close collaboration with the Managerial and Operational Center of the Amazon Protection System (CENSIPAM), aiming to integrate the results into the Fire Panel, a fire monitoring platform. Initially, using a dataset labeled by CENSIPAM, it was found necessary to expand the available features to ensure accurate classification through machine learning algorithms.

To achieve this goal, a methodological review was conducted, utilizing resources such as the GFED Amazon Dashboard, and collaborative consultations were held with CENSIPAM to identify essential characteristics for accurate categorization. Subsequently, two algorithms were evaluated: Random Forest and Multi-Layer Perceptron. The results obtained indicate an overall accuracy of around 77%, despite the challenges presented by the "Understory" and "Deforestation" classes, with the latter being of particular interest to CENSIPAM.

It is important to emphasize that **this repository contains exclusively non-sensitive content**, such as the codes used to acquire additional features and train the machine and deep learning algorithms. Given that **this work was carried out in direct collaboration between the University of Brasília (UnB) and the Brazilian Ministry of Defense, many data cannot be shared publicly**. The main purpose of this repository is to integrate it into the portfolio of the author of the work, Bruno Scholles Soares Dias.

<a name="det"></a>
## Project Detailing

### Data Acquisition/Extraction

In the `Feature_Extraction` folder, there is a Python code, developed in a Jupyter notebook, accompanied by the GeoPandas and RasterIO libraries. This code aims to add features to data coming from various GeoTIFFs. The added features include the Tree Cover Index, using data from Global Forest Change, biomass from GlobBiomass, and land cover from MapBiomas.

Additionally, overlay and intersection operations are conducted between shapefiles to incorporate information about indigenous areas, environmental conservation units, deforestation status, and land ownership, using data from the Rural Environmental Registry. These additions aim to enrich the understanding of patterns and impacts of forest fires.

Through these procedures, the goal is to provide a more comprehensive and detailed analysis of geospatial data related to forest fires, allowing for a better understanding of forest cover patterns, biomass, land cover, and their relationship with protected areas, deforestation, and land use.

### Training

The scripts `train-rf.py` and `train-mlp.py` represent, respectively, the training algorithms for Random Forest and Multi-Layer Perceptron (MLP). In the case of Random Forest, the developed code aims to classify fire events, utilizing the **scikit-learn library**. The initial configuration includes defining essential parameters, specifying datasets, and relevant features for event classification. After data preprocessing, the model is trained and statistically evaluated using metrics such as accuracy, recall, and f1-score to understand the model's performance.

For the MLP algorithm, training is conducted through code implemented using the **PyTorch library**. The MLP class plays a crucial role in structuring the machine learning model, creating a Multilayer Perceptron Neural Network to understand complex patterns in data and perform categorization tasks. The training process coordinates crucial steps such as data normalization, model training configuration, iteration over training and validation epochs, metric and result logging, as well as generating detailed reports for further analysis.

<a name="contact"></a>
## Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at brunoscholles98@gmail.com. Additionally, you can contact me via WhatsApp at +55 61 992598713.

<a name="thanks"></a>
## Thanks

Special thanks to my advisors [Edson Mintsu Hung](http://ft.unb.br/index.php?option=com_pessoas&view=pessoas&layout=perfil&id=143) and [Henrique Bernini](https://www.linkedin.com/in/henrique-bernini-01345735/?originalSubdomain=br).

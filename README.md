## Install dependencies

pip install -r requirements.txt
pip install -r requirements_gpu.txt

## Description of files

- All script files: Used to run the corresponding python files when making use of the Stanage hpc (they should be changed based on the hpc used).

- models.py: Contains the code for the models used in the project. 
- unzip_files.py: Used to unzip the files in the data folder (used to upload zipped files to hpc).
- myconfig.py: Creates needed directories and contains helper functions.
- augmentations.py: Contains the code for the augmentations used in the project.
- data_preprocessing.py: Contains the code for the data preprocessing used in the project.
- mri_model.py: Contains the code to train the MRI model used in the project.
- pet_model.py: Contains the code to train the PET model used in the project.
- final_with_logits.py: Performs uncertainty-based model selection using the logits of the models.
- final_with_sofftmax.py: Performs uncertainty-based model selection using the softmax of the models.
- accuracy_over_num_samples.py: Used to plot the accuracy over the number of samples used in the model selection.
- accuracy_in_uncertainty.py: Used to plot the accuracy at different uncertainty thresholds.

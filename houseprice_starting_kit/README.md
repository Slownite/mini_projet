This is a sample starting kit for the House price challenge. 


Prerequisites:
Install Anaconda Python 2.7, including jupyter-notebook

Usage:

(1) If you are a challenge participant:

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the houseprice challenge. At the prompt type:
  `jupyter-notebook README.ipynb`

- modify sample_code_submission to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run:

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly (you can substitute sample_xxx_data with any of the 4 kinds of sample data provided):

    `python ingestion_program/ingestion.py sample_xxx_data sample_result_submission ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_xxx_data sample_result_submission scoring_output`

- IMPORTANT: if you switch between sample data and public data, remove houseprice_model.pickle from sample_code_submission, otherwise you'll have inconsistent data and models.

- the metric identified by metric.txt in the utilities directory is the metric used both to compute performances in README.ipynb and for the challenge.
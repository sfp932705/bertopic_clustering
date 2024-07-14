# Content Clustering

Content clustering with BERTopic.

## Project Organization

```
├── Makefile             <- Makefile with convenience commands like `make format` or `make train`
│
├── README.md            <- The top-level README for developers using this project.
│
├── data                 <- Data artifacts folders.
│
├── pyproject.toml       <- Project configuration file with package metadata and configuration for
│                          tools like ruff.
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                  <- Source code for use in this project.
    │
    ├── data_processing  <- Data downloading and processing.
    │
    ├── modelling        <- Model training and inference.
    │
    └── tests            <- Tests directory
```

--------


Installation
------------
Create a working environment, for example with _venv_ and activate it.

Install the required modules `make install`.


Settings
--------
All project settings can be controlled from a _.env_ file or environment variables. Take
a
look [here](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to learn more.
An environment file which
overrides default values is provided as example.


Getting started
--------------

Once you have installed necessary modules, you can start experimenting.

As of now, only _.tsv_ files are supported. There are expected to be at least 2 columns:

* a content column
* a resume column that has the title/resume/summary of the content

Based on your data, change the name of these columns in settings.

By default, no preprocessing is performed. There are 2 others available:

* removing occurrences of 1 `@ @` or more
* replacing occurrences of 1 `@ @` or more with "[MASK]" and then filling in with
  BertForMaskedLM

It's simple to develop new custom preprocessing for your data.

First run the mlflow server to store experiment runs, with a command like this:

```shell
mlflow server --host 127.0.0.1 --port 5000
```

and then start training with

```shell
make train train-dataset=path_to_your_dataset_file
```

This will load settings and train models based on them. You can browse experiment
details with mlflow, and see each cluster created, as well as documents and topics in
them. Based on coherence scores, the model which makes best clusters will be downloaded
to your local file system for further use, for example like this:

```shell
make infer test-dataset=path_to_your_dataset_file model=saved_model_folder output=inferences_output_folder
```

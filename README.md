# Deepform

![Python build](https://github.com/project-deepform/deepform/workflows/Python%20application/badge.svg)
![Docker image](https://github.com/project-deepform/deepform/workflows/Docker%20Image%20CI/badge.svg)

Deepform is a project to extract information from TV and cable political advertising disclosure forms using deep learning.  [This public data, maintained by the FCC](https://publicfiles.fcc.gov/), is valuable to journalists but locked in PDFs.  Our goal is to provide the dataset for NLP/AI researchers and to make our method available to future data scientists working in this field.  Past projects have managed to produce similar data sets only with great manual effort or in a way which addresses only common form types, ignoring the tail of hundreds of rare form types.  This work uses models that are able to generalize over form types and "learn" how to find five fields:

- Contract number (multiple documents can have the same number as a contract for future air dates is revised)
- Advertiser name (offen the name of a political [comittee](https://www.fec.gov/data/browse-data/?tab=committees) but not always)
- Start and end air dates dates (often known as "flight dates")
- Total amount paid for the ads

The [initial attempt to use deep learning for this work](https://github.com/jstray/deepform), in summer 2019 by Jonathan Stray achieved 90% accuracy extracting total spending from the PDFs in the (held out) test set, which shows that deep learning can generalize surprisingly well to previously unseen form types.  For a discussion of how the 2019 prototype works, see [this post](http://jonathanstray.com/extracting-campaign-finance-data-from-gnarly-pdfs-using-deep-learning).

## Why?

TV stations are required to disclose their sale of political advertising, but there is no requirement that this disclosure is machine readable. Every election, tens of thousands of PDFs are posted to the FCC Public File, available at [https://publicfiles.fcc.gov/](https://publicfiles.fcc.gov/) in hundreds of different formats.

Past projects have used [volunteer labor](https://www.niemanlab.org/2012/12/crowdsourcing-campaign-spending-what-propublica-learned-from-free-the-files/) or [hand-coded form layouts](https://github.com/alexbyrnes/FCC-Political-Ads) to produce usable datasets.  Project Deepform replicates this data extraction using modern deep learning techniques.  This is desirable not only to produce a usable dataset in the context of the 2020 election (and all future US elections) but also as an open source springboard for future form extraction projects.  


## Running

The project is primarily intended to be run with [Docker](https://www.docker.com/products/docker-desktop), which eases issues with Python virtual environments, but it can also be run locally -- this is easiest to do with [Poetry](https://python-poetry.org/).

### Docker

To use Docker, you'll have to be running the daemon, which you can find and install from https://www.docker.com/products/docker-desktop. Fortunately, that's _all_ you need.

The project has a `Makefile` that covers most of the things you might want to do with the project. Run:

- `make test` to run all the unit tests for the project
- `make docker-shell` will spin up a container and drop you into a bash shell after mounting the `deepform` folder of code so that commands that you run there reflect the code as you are editing it.
- `make train` runs `deepform/train.py` with the default configuration. If it needs to it will download and preprocess the data it needs to train on.
- `make test-train` runs the same training loop on the same data, but with very strongly reduced settings (just a few documents for a few steps) so that it can be used to check that it actually works.
- `make sweep` runs a hyperparameter sweep with Weights & Biases, using the configuration in `sweep.yaml`

Some of these commands require an `.env` file located at the root of the project directory. It will need to have an API key you can get from your [settings](https://app.wandb.ai/settings) page at Weights & Biases. The file should look like:

```
WANDB_API_KEY=MY_API_KEY
```

If you don't want to use Weights & Biases, you can turn it off by setting `use_wandb=0`. You'll still need an `.env` file, but it can be empty.

#### Caveats

Training the model brings all the training data into memory and is quite RAM-intensive. On my 16GB machine, Docker will terminate with an out-of-memory exception if I train on more than approximately 6000 documents. If I train locally, it uses all my ram but will keep going after it exceeds it, slowing down and paging as it needs to. So with Docker, either train on a subset of the data (use a smaller `len_train`) or use a machine with a lot of RAM.


### Poetry - dependency management and running locally

Deepform manages its dependencies with `Poetry`, which you only need if you want to run it locally or alter the project dependencies.

You can install Poetry using any of the methods listed in their [documentation](https://python-poetry.org/docs/#installation).

If you want to run Deepform locally:

- run `poetry install` to install the deepform package and all of it's dependencies into a fresh virtual environment
- enter this environment with `poetry shell`
- or run a one-off command with `poetry run <command>`

Since deepform is an installed package inside the virtual environment Poetry creates, run the code as modules, e.g. `python -m deepform.train` instead of `python deepform/train.py` -- this insures that imports and relative paths work the way they should.

To update project dependencies:

- `poetry add <package>` adds a new python package as a requirement
- `poetry remove <package>` removes a package that's no longer needed
- `poetry update` updates all the dependencies to their latest non-conflicting versions

These three commands alter `pyproject.toml` and `poetry.lock`, which should be committed to git. Using them ensures that our project has reproducible builds.


## Creating the training data

### Summary
While all the data (training and test) for this project was originally raw PDFs, downloadable from the [FCC website](https://publicfiles.fcc.gov/) with up to 100,000 PDFs per election year.  The training data consists of some 20,000 of these PDFs, drawn from three different election years.  

The first components of the training data are three label manifests for these three election years (2012, 2014 and 2020), each of which contains a column of file IDs (called slugs) from that year and columns containing labels for each of the fields of interest for each document. The label manifests for 2012 and 2014 contain additional columns not used in this project.  

The second component of the training data is a set of approximately 20,000 .parquet files, one for each OCR'd PDF. The .parquet files are each named with the document slug and contain all of that document's tokens and their geometry on the page.  Geometry is given in 1/100ths of an inch.  

The .parquet files are formatted as "tokens plus geometry" like this:

`473630-116252-0-13442821773323-_-pdf.parquet` contains

```
page,x0,y0,x1,y1,token
0,272.613,438.395,301.525,438.439,$275.00
0,410.146,455.811,437.376,455.865,Totals
0,525.84,454.145,530.288,454.189,6
0,556.892,454.145,592.476,454.189,"$1,170.00"
0,18.0,480.478,37.998,480.527,Time
0,40.5,480.478,66.51,480.527,Period
...
```

The document name (the `slug`) is a unique document identifier, ultimately from the source TSV. The page number runs from 0 to 1, and the bounding box is in the original PDF coordinate system. The actual token text is reproduced as `token`. 

The total training data for project deepform consists of three label manifests and approximately 20,000 .parquet files containing tokens. 

### Acquiring the Three Label Manifests
#### 2012 Label Manifest
In 2012, ProPublica ran the Free The Files project (you can [read how it worked](https://www.niemanlab.org/2012/12/crowdsourcing-campaign-spending-what-propublica-learned-from-free-the-files/)) and hundreds of volunteers hand-entered information for over 17,000 of these forms. That data drove a bunch of campaign finance [coverage](https://www.propublica.org/series/free-the-files) and is now [available](https://www.propublica.org/datastore/dataset/free-the-files-filing-data) from their data store.

The label manifest for 2012 data was downloaded from Pro Publica and is located at `data/2012_manifest.tsv` (renamed from ftf-all-filings.tsv which is the filename it downloads as).  If the manifest is not present, it can be recovered from [their website](https://www.propublica.org/datastore/dataset/free-the-files-filing-data). This file contains the crowdsourced answers for some of our targets (omitting flight dates) and the PDF url.

#### 2014 Label Manifest
In 2014 Alex Byrnes [automated](https://github.com/alexbyrnes/FCC-Political-Ads) a similar extraction by hand-coding form layouts. This works for the dozen or so most common form types but ignores the hundreds of different PDF layouts in the long tail. 

The label manifest for 2014 data, acquired from Alex's Github is `data/2014_manifest.tsv`.  If the manifest is not present, it can be recovered from [his github](https://github.com/alexbyrnes/FCC-Political-Ads) (renamed from 2014-orders.tsv which is the filename it downloads as). This file contains the crowdsourced answers for some of our targets (omitting 'gross amount').


#### 2020 Label Manifest

##### All 2020 PDFs
Pdfs for the 2020 political ads and associated metadata were uploaded to [Overview Docs](https://www.overviewdocs.com/documentsets/22569). To collect the pdfs, the file names were pulled from the [FCC API OPIF file search](https://publicfiles.fcc.gov/developer/) using the search terms: order, contract, invoice, and receipt. The search was run with filters for campaign year set to 2020 and source service code set to TV. 

The FCC API search also returns the source service code (entity type, i.e. TV, cable), entity id, callsign, political file type (political ad or non-candidate issue ad), office type (presidential, senate, etc), nielsen dma rank (tv market area), network affiliation, and the time stamps for when the ad was created and last modified were pulled. These were added to the overview document set along with the search term, URL for the FCC download, and the date of the search.

For these .pdfs, the following steps were followed to produce training data:

 - Convert pdf to a series of images
 - Determine angle of each page and rotate if needed
 - Use tesseract to OCR each image
 - Upload processed pdf to an S3 bucket and add URL to overview
 - Upload additional metadata on whether OCR was needed, the original angle of each page, and any errors that occurred during the OCR process.  

##### A Subset for Training
[A sample of 1000 documents](https://www.overviewdocs.com/documentsets/22186) were randomly chosen for hand labeling as 2020 training data.  

The label manifest for 2020 data is `data/2020_manifest.tsv` (renamed from fcc-data-2020-sample-updated.csv which is the filename it downloads as).  If the manifest is not present, it can be recovered from [this overview document set](https://www.overviewdocs.com/documentsets/22186). This file contains our manually entered answers for all of our five targets for the 1000 randomly chosen documents.


### Acquiring the PDFs or Token Files
#### Acquiring .parquet files directly

The best way to run this project is to acquire the 20,000 .parquet files containing the tokens and geometry for each PDF in the training set. The token files are downloaded from our S3 bucket by running `make data/tokenized`.  These .parquet files are then located in the folder data/tokenized.  This is the easiest way to get this data.  

#### Acquiring Raw PDFs

To find the original PDFs, it is always possible to return to the [FCC website](https://publicfiles.fcc.gov/) and download them directly using the proper filters (search terms: order, contract, invoice, and receipt, filters:  campaign year = 2020, source service code = TV).  Each of the 2012, 2014 and 2020 data which was used by Pro Publica, by Alex Byrnes or by ourselves to create the three label manifests can be found in a differnt location also as follows: 

##### 2012 Training PDFs

The original PDFs from the Free the Files Project are available on DocumentCloud or in this [folder](https://drive.google.com/drive/folders/1bsV4A-8A9B7KZkzdbsBnCGKLMZftV2fQ?usp=sharing). 

##### 2014 Training PDFs

[Alex Byrnes github](https://github.com/alexbyrnes/FCC-Political-Ads) directs users back to the [FCC website](https://publicfiles.fcc.gov/) to get his data.  He does not host it separately.  

##### 2020 Training PDFs

The one thousand 2020 PDFs we hand labeled are available on Overview Docs as [this dataset](https://www.overviewdocs.com/documentsets/22186) 

### Combining and Peparing the Data 

- A vocabulary of the tokens and their frequencies is created by running (if using docker) `make data/token_frequency.csv` or simply (if using poetry) `python -m deepform.data.create_vocabulary`. 
- The three manifests should be present in the data folder.  If they are not, they can be downloaded from the three data sources as detailed above. 
- The three individual manifests are combined into one by running (if using docker) `make data/3_year_manifest.csv` or (if using poetry) `python -m deepform.data.combine_manifests`. This combined manifest includes a column 'year' so that training data drawn from the three years can be balanced for various purposes.  
- The tokenized data (the .parquet files) are prepared for model training by running (if using docker) `make data/doc_index.parquet` or simply (if using poetry) `python -m deepform.data.add_features data/3_year_manifest.csv`.  This script adds a column to the token file for each of the five target types.  This column is used to store the match percentage, for each token between that token and the target in question.  Some targets are more than one token in length so in these cases, this new column contains the likelihood that each token is a member of the target token string.  This script also computes other relevant features such as whether the token is a date or a dollar amount which are fed into the model as additional features.  
- Having created the three-year manifest, downloaded the token files and run `make data/doc_index.parquet`, the model is ready to train.  



## How the model works

The easiest fields are contract number and total. This uses a fully connected three-layer network trained on a window of tokens from the data, typically 20-30 tokens. Each token is hashed to an integer mod 1000, then converted to 1-hot representation and embedded into 64 dimensions. This embedding is combined with geometry information (bounding box and page number) and also some hand-crafted "hint" features, such as whether the token matches a regular expression for dollar amounts. For details, see [the talk](https://www.youtube.com/watch?v=uNN59kJQ7CA).

We also incorporate custom "hint" features. For example, the total extractor uses an "amount" feature that is the log of the token value, if the token string is a number.


## Code quality and pre-commit hooks

The code is currently automatically formatted with [black](https://black.readthedocs.io/en/stable/), uses [autoflake](https://pypi.org/project/autoflake/) to remove unused imports, [isort](https://timothycrosley.github.io/isort/) to sort them, and [flake8](https://flake8.pycqa.org/en/latest/) to check for PEP8 violations. These tools are configured in `pyproject.toml` and should Just Work&trade; -- you shouldn't have to worry about them at all once you install them.

To make this as painless as possible, `.pre-commit-config.yaml` contains rules for automatically running these tools as part of `git commit`. To turn these git pre-commit hook on, you need run `pre-commit install` (after installing it and the above libraries with Poetry or pip). After that, whenever you run `git commit`, these tools will run and clean up your code so that "dirty" code never gets committed in the first place.

GitHub runs a "python build" Action whenever you push new code to a branch (configured in [python-app.yml](https://github.com/project-deepform/deepform/blob/master/.github/workflows/python-app.yml)). This also runs `black`, `flake8`, and `pytest`, so it's best to just make sure things pass locally before pushing to GitHub.

## A research data set

This is a difficult data set that is very relevant to journalism, and improvements in technique will be immediately useful to campaign finance reporting.

The general problem is known as "knowledge base construction" in the research community, and the current state of the art is achieved by multimodal systems such as [Fonduer](https://fonduer.readthedocs.io/en/latest/).

We would love to hear from you! Contact jstray on [twitter](https://twitter.com/jonathanstray) or through his [blog](http://jonathanstray.com).

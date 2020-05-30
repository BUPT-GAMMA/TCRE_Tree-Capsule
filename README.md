Code for the under-reviewed EMNLP 2020 paper "Tree-Capsule: Tree-Structured Capsule Network for Improving Relation Extraction".  For convenience, we update and complete the code based on the https://github.com/thunlp/OpenNRE


Make sure the following files are present as per the directory structure before running the code,

```
TreeCapsule
├── benchmark
│   ├── semeval
│   │   ├── semeval_rel2id.json
│   │   ├── semeval_test.txt
│   │   ├── semeval_train.txt
│   │   └── semeval_val.txt
│   └── *.py
├── opennre
│   └── *
├── pretrain
│   ├── download_glove.sh
│   ├── semeval_lstmCat.pth.tar
│   └── tacred_lstmpa.pth.tar
├── *.py
├── README.md
└── requirements.txt
```

For TACRED dataset, You need to first download dataset from [LDC](https://catalog.ldc.upenn.edu/LDC2018T24), which due to the license issue I cannot put in this repo. 



# Dependencies

Our code runs on the CentOS Linux release 7.7.1908 (Core) with GeForce RTX 2080 Ti (11GB) device, and the following packages installed: 

```
python 3.7

boto3==1.10.2
botocore==1.13.2
certifi==2019.9.11
chardet==3.0.4
Click==7.0
docutils==0.15.2
idna==2.8
jmespath==0.9.4
joblib==0.14.0
numpy==1.17.3
python-dateutil==2.8.0
regex==2019.8.19
requests==2.22.0
s3transfer==0.2.1
sacremoses==0.0.35
scikit-learn==0.21.3
scipy==1.3.1
sentencepiece==0.1.83
six==1.12.0
sklearn==0.0
torch==1.3.1
tqdm==4.36.1
transformers==2.8.0
urllib3==1.25.6
nltk==3.2.5
```



# Run

Train and test,

```
python main.py
```

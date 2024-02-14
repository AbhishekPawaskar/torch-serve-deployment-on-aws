# CI/CD pipeline for building Deployable Deep Learning Models in AWS using Torch Serve (PyTorch)

Detailed Documentation to Setup CI/CD pipeline using AWS for Deep Learning Models in Kubernetes.

## OverView




## Contents

1. Requirements
2. Setup
3. Triggering Deployments
4. FAQ

## 1. Requirements

1. AWS Account
2. GitHub Repository
3. Deep Learning Model (PyTorch Model)

## 2. Setup

#### A. GitHub Repository: 
Make Sure you have following files in the root of the repository:
    
    > Dockerfile
    
    > buildspec.yml
    
    > deployments.yml


##### Dockerfile:

Standard `Dockerfile` for Building Images of Servable Deep Learning Models using PyTorch is as follows:


```bash

FROM pytorch/torchserve:0.9.0-cpu

ENV MODEL_NAME=ENTER_MODEL_NAME_HERE

WORKDIR /home/model-server/

COPY ${MODEL_NAME}.mar .

CMD ["torchserve", "--start", "--model-store", "/home/model-server", "--models", "model=<MODEL_NAME>.mar"]

```


##### buildspec.yml:

Standard `buildspec.yml` for getting the models form `AWS S3` and  Building Containers and inserting the container in `AWS ECR` is as follows (You may add image tags if you want to):


```bash
version: 0.2

phases:
  pre_build:
    commands:
      - aws s3 cp $S3_BUCKET_LINK ./$MODEL_NAME --recursive
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
  
  build:
    commands:
      - docker build -t $IMAGE_REPO_NAME .
      - docker tag $IMAGE_REPO_NAME $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME
  
  post_build:
    commands:
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME
```

#### B. AWS S3: 

a. Save the PyTorch model using `jit.trace` method
```bash

import torch

model = YourPyTorchModel()

traced_model = torch.jit.trace(model, torch.rand(1, input_size))

traced_model.save("model_name.pt")

```

b. Create a Handler `model_handler.py` (Sample handler given below for further customization)
```bash
# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import os
import torch
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        #  load the model, refer 'custom handler class' above for details
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        
        request_body = data[0].get("body")
        received_data = request_body.get("input_data")
        refined_input = torch.tensor([received_data])
        return refined_input


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output.tolist()
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

```

c. Create a `.mar` file using `torch-model-archiver` as follows:
```bash

$ pip install torch-model-archiver

$ torch-model-archiver --model-name <MODEL_NAME> --version 1.0 --serialized-file <MODEL_NAME>.pt --handler model_handler.py --export-path ./

```

d. Create a S3 bucket in AWS and Upload the `<MODEL_NAME>.mar`  file in the bucket. You may refer the [Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/upload-objects.html)


#### C. AWS ECR:

a. Navigate to `AWS Elastic Container Registry` using the Search option in the `AWS Console`

b. Under `Private Repository`, Click `Create Repository` option.

c. Give the name of choice and create. (Rest of the Settings are not complusory and you may choose defaults unless you need something to be specific.)


#### D. AWS CodeBuild:

a. Navigate to AWS CodeBuild using the Search option in the AWS Console

b. Under `build projects`, look for `Create Build Project` option.

c. Enter Project Name of choice and under `Source` choose `GitHub` as the source.

d. After choosing `GitHub` as the option you'll observe an option to connect AWS account with your GitHub's account. (One time thing, not necessary if already done.) You can either use `Personal Access Tokens` or `oAuth` to connect AWS & GitHub.

e. Choose the Repository consisting of corresponding Dockerfile & buildspec.yml files.

f.  Rest of the Settings are not complusory and you may choose defaults unless you need something to be specific.

g. click on `create build project`.

h. Revisit this Build Project and add Environment Variables required/mentioned in the `buildspec.yml` file. It is not a good idea to have sensitive information like account id in and `.env` file in the GitHub Repository. 


#### E. AWS CodePipeline:

a. Navigate to AWS CodePipeline using the Search option in the AWS Console

b. Under `pipelines`, look for `Create new pipeline` option.

c. Enter Pipeline Name of choice and under `Pipeline type` choose `V2` as the type. Click on `Next`. (unless you may choose defaults unless you need something to be specific.)

d. Under Source, Choose `GitHub Version 2` and then click on connect to GitHub. Here you may choose existing connection or Create a new on the spot.

e. Choose the Repository and choose how do you want to trigger the pipeline. If you want to trigger the pipeline when there is A `PUSH` to the `MAIN/MASTER` branch, then choose `PUSH` and choose branch name as `main`. Clicke `Next`.

f. Under Build Stage Choose the CodeBuild Project you created in previous step. and click `Next` and then another `Next`. (Ignore Stage option for now).

g. Once the pipeline is created it will start running and as default it will Fail, which is the intended result. 


#### F. AWS IAM:

a. Navigate to AWS IAM using the Search option in the AWS Console.

b. Under Policies, choose the Policies automatically created by codebuild & CodePipeline for their usage respectively. check for the necessary permission in the policies.

c. The policy for CodeBuild must have access to read the S3 bucktes and have read & right access to AWS ECR anf read access to get Read GitHub repo contents as well. Use the Visual Editor to enable the permissions in the Policies.

## 3. Triggering Deployments

There are two major Parts to this.

1. Creating the Container.
2. Deploying the Container in Kubernetes.

1. Creating the Container: This pipeline can be triggered using a git push/commit in the `main` branch or can be manually triggered by using `Run Build` or `Run pipeline` Options in `AWS CodeBuild` & `AWS CodePipeline` respectively.

2. Deploying the Container in Kubernetes: You can achive this by having Deployments pipeline similar to that of Backend Deployments and also using External tools like ArgoCD etc. But if would like to manually deploy in Kubernetes, then you may use the `Deployments.yaml` to deploy the same. 

```bash
#1. Navigate to location where Deployments.yaml file exists
#2. Authenticate and connect to your cluster
#3. deploy using the command

$ kubectl create -f Deployment.yaml

```

# Running on AzureML

## Steps
1. Follow the instructions for pre-processing the data
2. Install the requirements for the submission script `pip install -r requirements.txt`
3. Create an [AzureML workspace](https://azure.microsoft.com/en-us/services/machine-learning/). Obtain the subscription id, resource group and workspace name from the azure portal. Initialize the submit script with these details
```
./submit.py init --subscription_id SUBSCRIPTION_ID
                 --resource_group RESOURCE_GROUP
                 --workspace_name WORKSPACE_NAME
```
4. Launch the AML workspace portal from the azure portal. Go to Datastores -> workspaceblobstore. Note down the name of the Account name and Blob container. These are created along with the workspace
5. Install and launch [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/). Sign in with your account and select the subscriptions that you used for the AML workspace
6. On the right panel, navigate to the storage account found in step 4 and open the blob container from step 4
7. Create folders named Code and Data, and put the contents of the Code directory and Processed Data directory in them respectively
8. You can upload any custom bert based models for evaluation to the blob container. Create a folder named Models and put in each model you want in a new directory inside it. Edit the BERT_MODEL variable in the YAML file to point to this
9. Back in the AML workspace portal, go to Compute ->  Compute Clusters and create a new single GPU per node training cluster, i.e one of NV6/NC6sv2/NC6sv3/ND6s (M60/P100/V100/P40). Note down the cluster name and fill this in the supplied YAML file
10. To submit a run for all the tasks in GLUECoS, run
```
./submit.py run gluecos.yaml run0
```
Here `run0` is a name given to this set of runs. You can give any name that you want
To submit a run for certain tasks alone, use the -j flag and specify the task names (from the YAML file) that you want to run
```
./submit.py run gluecos.yaml run1 -j en_hi_sentiment en_es_sentiment
```
After submitting, you'll get a URL to the Azure portal where you can view some stats and the STDOUT of the run

11. You can use the status command to monitor the status of each submission
```
./submit.py status run0
```
12. Once the runs have completed, you can navigate to the output directories via Storage Explorer to view the outputted labels for the test set. The exact location will be
```
blob_container_name -> Experiments -> run0 (the name you gave while submitting) -> en_hi_sentiment (the task name)
```
13. You can edit the runs in the YAML file to use different models (bert/xlm-roberta) or change any other parameters
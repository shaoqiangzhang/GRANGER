# GRANGER 
(Granger Recurrent Autoencoders to Network Gene Expression Regulations): Gene Regulatory Network Inference from Time-series scRNA-seq Data via Granger Causal Recurrent Autoencoders.

## Requirements
GRANGER requires you to create a PyTorch environment and install the following packages

python                    3.11.5 

pytorch                   2.1.0

pandas                    2.1.4

scanpy                    1.10.1

scikit-learn              1.2.2

numpy                     1.26.0 


## Scripts
### The code of GRANGER mainly includes the following scripts:

**models/granger_model.py**: the core code of the GRANGER model. A Granger Causal(GC) matrix will be generated and saved as "GC_cell.csv". 

**models/utils.py**: define some functions for obtaining list of transcription factors (tf_list) and gene expression matrix, and formating label files and calculating AUROC and AUPRC scores.

**generate_npy.py**: this script is used to process data with time-series information, which can reorder cells in "ExpressionData.csv" according to the pseudotime information in "PseudoTime.csv" and generate a new numpy file (.npy). For data without time  information, you can use "to_npy" to directly generate an input file.

**GRANGER_demo.py**:a demo to run the model with modifiable parameters (you can change the parameters to get better results).

**format_result.py**: format the "GC_cell.csv" generated by the model to obtain a gene-pair format with regulatory information (includes cause gene, effect gene, regulate or not, regulatory_score, sign, etc).

## Usage
### Step 1: preprocess data into .npy format 
Case 1: If you use the ".npy" files we provided in the example_data directory, you can directly use the file to run our model. 

Case 2: If you use a dataset from BEELINE, which contains two csv files (ExpressionData.csv and PseudoTime.csv), 
you can use "generate_npy.py" to merge ExpressionData.csv and PseudoTime.csv and transfer into .npy format (save as "time_output.npy").

Case 3: If you want to use your own dataset, you are required to use "scanpy_paga.py" to preprocess your dataset and predict pseudotime to obtain a .npy file. 

### Step 2: run a demo
Modify the file path in GRANGER_demo.py and models/granger_model.py to yours, and then run GRANGER_demo.py directly.
If you want to use unlabeled data, you need to comment out the use of utils.py in models/granger_model.py.

You can also adjust some parameters to get better results before run the demo.  


### Step 3: format the result
Use format_result.py to convert the "GC_cell.csv" into a gene-pair format with regulatory information and save as "result.csv".  

### Note:
If you input too many genes or certain parameters, it may consume a large amount of GPU memory, causing memory overflow and interrupting the process.

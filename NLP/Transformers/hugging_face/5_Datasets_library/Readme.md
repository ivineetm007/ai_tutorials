# Datasets library

## Working with local and remote datasets
🤗 Datasets provides loading scripts to handle the loading of local and remote datasets. It supports several common data formats, such as: csv/tsv, text files, json, pickled dataframes etc.

**Local dataset**
To load a local dataset pass the dataset pass the path to the files and the file format. Datasets also supports automatic decompression of the input files.
```
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```
**Remote dataset**
 Instead of providing a path to local files, we point the data_files argument of load_dataset() to one or more URLs where the remote files are stored. 
```
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

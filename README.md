#TODO

```bash
# Prep data
mkdir -p lists
find aclImdb_v1/aclImdb/train/neg -mindepth 1 | head -n 8000 > lists/train_neg.txt
find aclImdb_v1/aclImdb/train/neg -mindepth 1 | tail -n 4500 > lists/validation_neg.txt
find aclImdb_v1/aclImdb/train/pos -mindepth 1 | head -n 8000 > lists/train_pos.txt
find aclImdb_v1/aclImdb/train/pos -mindepth 1 | tail -n 4500 > lists/validation_pos.txt
find aclImdb_v1/aclImdb/test/neg -mindepth 1 > lists/test_neg.txt
find aclImdb_v1/aclImdb/test/pos -mindepth 1 > lists/test_pos.txt

# Install
pip install -r requirements.txt
python install.py

#Run example
python example.py

```




#DOKU
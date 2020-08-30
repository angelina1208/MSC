#TODO

```bash
# Maybe you want to use a venv
python3.8 -m venv venv/
source venv/bin/activate

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# Prepare data into train validation and test sets
mkdir -p lists
find aclImdb/train/neg -mindepth 1 | head -n 8000 > lists/train_neg.txt
find aclImdb/train/neg -mindepth 1 | tail -n 4500 > lists/validation_neg.txt
find aclImdb/train/pos -mindepth 1 | head -n 8000 > lists/train_pos.txt
find aclImdb/train/pos -mindepth 1 | tail -n 4500 > lists/validation_pos.txt
find aclImdb/test/neg -mindepth 1 > lists/test_neg.txt
find aclImdb/test/pos -mindepth 1 > lists/test_pos.txt

# Install
pip install -r requirements.txt
python src/MSC_install.py
pytest test/test_MSC.py

#Run example
mkdir -p data
python example.py

```




#DOKU
import os

# do your stuff

if __name__ == '__main__':
	print("Welcome to MSC.")
	pos_folder = "aclImdb_v1/aclImdb/train/pos"
	neg_folder = "aclImdb_v1/aclImdb/train/neg"
	print("Training with positive samples from: {}\nand negative samples from: {}".format(pos_folder, neg_folder))
	for filename in os.listdir(pos_folder):
		with open(os.path.join(os.getcwd(), pos_folder,  filename), 'r') as f:
			text = f.read()

		print(f.read())

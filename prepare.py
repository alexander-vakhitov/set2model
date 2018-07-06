import lmdb_filler as filler

db_folder = '<Folder For Database>'
img_folder = '<Folder for Images>'
filler.prepare_lmdb_database(db_folder, img_folder, 'lmdb_flower_train', 'train.txt')
filler.prepare_lmdb_database(db_folder, img_folder, 'lmdb_flower_test', 'test.txt')
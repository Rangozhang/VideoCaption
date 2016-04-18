require 'misc.DataLoader'

local loader = DataLoader{h5_file = '', json_file = '', frame_length = 10}

print(loader:getSeqLength())

print(loader:getVocabSize())

print(loader:getVocab())

print(loader:getBatch{batch_size = 16, split = 'train', seq_per_img = 5})

print(loader:resetIterator('train'))

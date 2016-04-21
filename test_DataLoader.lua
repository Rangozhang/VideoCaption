require 'misc.DataLoader'

local loader = DataLoader{h5_file = 'full_test_data_real.h5', json_file = 'full_test_data.json', frame_length = 10}
print('init Dataloader done....\n\n\n')
--print(loader:getSeqLength())
print('get seqlength done....\n\n\n')
--print(loader:getVocabSize())
print('get Vocal Size done...\n\n\n')
--print(loader:getVocab())
print('get whole Vocab done...\n\n\n')
print(loader:getBatch{batch_size = 16, split = 'train', seq_per_img = 5, frames_per_video = 10})
print('get  Batch done....finally......\n\n\n')
print(loader:resetIterator('train'))
print('donkey is ready to ship!!!!')

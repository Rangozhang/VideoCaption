require 'misc.DataLoaderRaw'

local loader = DataLoaderRaw{ folder_path = '/home/rz1/VisualLearningProj/dataset/parsed_dataset/YouTubeClips/4UOVKok7j1U_1_8', prefix = '4UOVKok7j1U_1_8'}
print('init Dataloader done....\n\n\n')
--print(loader:getSeqLength())
print('get seqlength done....\n\n\n')
--print(loader:getVocabSize())
print('get Vocal Size done...\n\n\n')
--print(loader:getVocab())
print('get whole Vocab done...\n\n\n')
print(loader:getBatch{batch_size = 1})
print('get  Batch done....finally......\n\n\n')
print(loader:resetIterator('train'))
print('donkey is ready to ship!!!!')

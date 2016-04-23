import json
f = open('full_test_data.json')
data = json.load(f)
data = data['images']
all_list = list()
for i in data:
    if i['split'] == 'val':
        all_list.append(i['file_path'])
        print i['file_path']

caption_list = dict()
f = open('../dataset/msvd_caption.json')
data = json.load(f)
for i in data:
    caption_list[i['file_path']] = i['captions']

output_file = list()
for i in all_list:
    caption_ = caption_list[i]
    for x in caption_:
        output_file.append({'image_id':i, 'caption':x})
        print(i + ': '+ x)

json.dump(output_file, open('val_labels.json', 'wb'))
print('DONE!!!')

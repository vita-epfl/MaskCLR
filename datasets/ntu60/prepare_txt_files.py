import pickle
from tqdm import tqdm

with open('Frames/ntu60_hrnet.pkl', 'rb') as f:
    x = pickle.load(f)




t = open('data_fixed/train_videofolder_retrieved.txt', 'w')

v = open('data_fixed/val_videofolder_retrieved.txt', 'w')

a = open('data_fixed/all.txt', 'w')


for ann in tqdm(x['annotations']):
    line = ann['frame_dir'] + ' ' + str(ann['total_frames']) + ' ' + str(ann['label'])

    a.write("%s\n" % line)

    if ann['frame_dir'] in x['split']['xsub_train']:
        t.write("%s\n" % line)
    elif ann['frame_dir'] in x['split']['xsub_val']:
        v.write("%s\n" % line)
    
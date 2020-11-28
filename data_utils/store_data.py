import pickle, json, h5py
import numpy as np
import os, argparse

def check_exists(path):
    return os.path.exists(path)

def get_coco_img_id(src_cap,coco_cap_dict):
    tar_img_id = ''
    for annot in coco_cap_dict['annotations']:
        img_id = annot['image_id']
        cap = annot['caption']
        if cap == src_cap:
            tar_img_id = img_id
            break
    return tar_img_id

def store_2d3d(out_path):
    pre_path = '/data/datasets/vid_tx_data/youcookv2/'
    annot_path = pre_path+'youcookii_caption_transcript.pickle'
    raw_annot_path = pre_path+'youcookii_annotations_trainval.json'
    coco_cap_path = pre_path+'val_caption_coco_format.json'
    feats_2d_pre_path = pre_path+'feature.2d/'
    feats_3d_pre_path = pre_path+'feature.3d/'
    target_split = 'validation'

    annot_dict = pickle.load(open(annot_path, 'rb'))
    raw_annot_dict = json.load(open(raw_annot_path, 'r'))
    coco_cap_dict = json.load(open(coco_cap_path, 'r'))

    data_dict = {}
    faulty = set()
    for cnt,vid_id in enumerate(annot_dict):
        split = raw_annot_dict['database'][vid_id]['subset']
        if split != target_split:
            continue

        feats_2d_path = '{}{}_features.npy'.format(feats_2d_pre_path,vid_id)
        feats_3d_path = '{}{}_features.npy'.format(feats_3d_pre_path,vid_id)
        if (not check_exists(feats_2d_path)) or (not check_exists(feats_3d_path)):
            faulty.add(vid_id)
            continue
        
        feats_2d = np.load(feats_2d_path)
        feats_3d = np.load(feats_3d_path)
        num_segs = len(annot_dict[vid_id]['text'])
        seg_dict = {'{}'.format(n): {} for n in range(num_segs)}
        for n in range(num_segs):
            start = annot_dict[vid_id]['start'][n]
            end = annot_dict[vid_id]['end'][n]
            idx_3d = [i for i in range(int(np.floor(start * 1.5)), int(np.ceil(end * 1.5)))]
            idx_2d = [int(float(i) / 1.5) for i in idx_3d]
            feat_2d = feats_2d[idx_2d]
            feat_3d = feats_3d[idx_3d]

            caption = annot_dict[vid_id]['text'][n]
            transcript = annot_dict[vid_id]['transcript'][n]
            coco_img_id = get_coco_img_id(caption, coco_cap_dict)
            features = np.concatenate([feat_2d, feat_3d], 1)
            seg_dict[str(n)] = {'image_id': coco_img_id,
                                'caption': caption,
                                'transcript': transcript,
                                'features': features}

        data_dict[vid_id] = seg_dict
        print('{}/{}'.format(cnt+1,len(annot_dict)))

    save_path = os.path.join(out_path, 'youcook2/features_2d3d/Res152-S3D_Jianwei_{}.pkl'.format(target_split))
    pickle.dump(data_dict, open(save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Storing finished with {} faulty features.'.format(len(faulty)))

def generate_coco_format_captions(cap_dict):
    coco_dict = {'type': 'captionns', 
                 'info': 'dummy', 
                 'licenses': 'dummy', 
                 'images': [],
                 'annotations': []}

    for vid_id in cap_dict:
        for seg_id in cap_dict[vid_id]:
            image_id = cap_dict[vid_id][seg_id]['image_id']
            caption = cap_dict[vid_id][seg_id]['caption']
            caption_id = int(image_id)
            coco_dict['images'].append({'id': image_id, 'file_name': image_id})
            coco_dict['annotations'].append({"image_id": image_id, "caption": caption, "id": caption_id})

    return coco_dict

def generate_coco_format_multi_ref_captions(cap_dict):
    coco_dict = {'type': 'captionns', 
                 'info': 'dummy', 
                 'licenses': 'dummy', 
                 'images': [],
                 'annotations': []}

    seen_images = set()
    for vid_id in cap_dict:
        for seg_id in cap_dict[vid_id]:
            image_id = cap_dict[vid_id][seg_id]['image_id']
            caption = cap_dict[vid_id][seg_id]['caption']
            caption_id = cap_dict[vid_id][seg_id]['caption_id']
            coco_dict['annotations'].append({"image_id": image_id, "caption": caption, "id": caption_id})
            if image_id not in seen_images:
                seen_images.add(image_id)
                coco_dict['images'].append({'id': image_id, 'file_name': image_id})
            

    return coco_dict

def append_pos_ids(features):
    T, H, W, D = features.shape
    frame_ids = np.array(range(T))[:, np.newaxis, np.newaxis, np.newaxis]
    frame_ids = np.tile(frame_ids, [1, H, W, 1])
    pos_ids = np.zeros((1, H, W, 4), dtype='float32')
    for y in range(H):
        for x in range(W):
            pos_ids[0,y,x,:] = [x/W, (x + 1)/W, y/H, (y + 1)/H]
    pos_ids = np.tile(pos_ids, [T, 1, 1, 1])
    features_pos = np.concatenate([features,frame_ids,pos_ids], axis=3)

    return features_pos

def store_activity_net(in_path, out_path, annot_path, avg_pool):
    splits_ids_dict = json.load(open(os.path.join(annot_path, 'splits_ids.json'), 'r'))
    caps_dict_train = json.load(open(os.path.join(annot_path, 'captions', 'train.json'), 'r'))
    caps_dict_val_1 = json.load(open(os.path.join(annot_path, 'captions', 'val_1.json'), 'r'))
    caps_dict_val_2 = json.load(open(os.path.join(annot_path, 'captions', 'val_2.json'), 'r'))
    caps_dict = {'training': caps_dict_train, 'validation_1': caps_dict_val_1, 'testing_1': caps_dict_val_1,
                                              'validation_2': caps_dict_val_2, 'testing_2': caps_dict_val_2,
                }
    
    cnt_dict = {'training': -1, 'validation_1': -1, 'testing_1': -1,
                                'validation_2': -1, 'testing_2': -1}

    num_chunks = {'training': 128, 'validation_1': 64, 'testing_1': 64,
                                   'validation_2': 64, 'testing_2': 64}

    cap_2_feat_split = {'training': 'training', 'validation_1': 'validation', 'testing_1': 'validation',
                        'validation_2': 'validation', 'testing_2': 'validation'}

    n_all_vids = sum([len(splits_ids_dict[split]) for split in splits_ids_dict])
    cnt_vid = 0
    faulty = set()
    for split in splits_ids_dict:
        #prepare per_split variables
        per_split_n_vids = len(splits_ids_dict[split])
        per_chunk_n_vids = int(per_split_n_vids/num_chunks[split]) + 1
        per_split_cnt_vid = 0
        per_split_cnt_chunk = 0
        base_dir_path = os.path.join(out_path, split)
        os.makedirs(base_dir_path, exist_ok=True)

        cap_dict = {}
        hf = None
        for vid_id in splits_ids_dict[split]:
            per_split_cnt_vid += 1

            if (per_split_cnt_vid % per_chunk_n_vids == 0) or (hf is None):
                if hf is not None:
                    hf.close()
                hf = h5py.File(os.path.join(base_dir_path, 'data_part_{}.h5'.format(per_split_cnt_chunk)), 'w')
                per_split_cnt_chunk += 1

            cnt_vid += 1
            feat_path = os.path.join(in_path,cap_2_feat_split[split],vid_id+'.npy')
            if not check_exists(feat_path):
                faulty.add(vid_id)
                continue
            features = np.load(feat_path)

            annots = caps_dict[split][vid_id]
            seg_dict = {}
            n_annots = len(annots['sentences'])
            for n in range(n_annots):
                seg_id = str(n)

                timestamp = annots['timestamps'][n]
                caption = annots['sentences'][n]
                
                start = int(np.floor(timestamp[0]))
                end = int(np.ceil(timestamp[1]))
                
                start = max(start-1, 0) #more coverage
                end = min(end,features.shape[0]) #cap to max feature length

                if start >= end:
                    continue

                seg_features = features[start:end,:]
                if avg_pool:
                    seg_features = np.average(seg_features, axis=(1,2))
                else:
                    seg_features = append_pos_ids(seg_features)
                    seg_features = np.reshape(seg_features, [-1, seg_features.shape[-1]])

                cnt_dict[split]+=1
                image_id = str(cnt_dict[split])

                hf.create_dataset('/'.join([vid_id, seg_id, 'image_id']), data=image_id)
                hf.create_dataset('/'.join([vid_id, seg_id, 'caption']), data=caption)
                hf.create_dataset('/'.join([vid_id, seg_id, 'features']), data=seg_features)

                seg_dict[seg_id] = {'image_id': image_id,
                                    'caption': caption}

            if len(seg_id) == 0:
                faulty.add(vid_id)
                continue

            cap_dict[vid_id] = seg_dict

            print('{}/{}'.format(cnt_vid, n_all_vids))

        hf.close()
        print('\nSplit "{}" done.\nGenerating COCO-format captions...'.format(split))
        coco_format_caps = generate_coco_format_captions(cap_dict)
        save_path = os.path.join(base_dir_path, 'coco_format_caps.json'.format(split))
        json.dump(coco_format_caps, open(save_path, 'w'))

        print('Resuming...\n\n')

    print('Storing finished with {} faulty features.'.format(len(faulty)))

def store_youcook2(in_path, out_path, annot_path, avg_pool):
    annot_dict_path = os.path.join(annot_path, 'youcookii_annotations_trainval.json')
    annot_dict = json.load(open(annot_dict_path, 'r'))['database']
    
    #split annotation dictionary based on data_split_type
    splitted_annot_dict = {'training': {}, 'validation': {}}
    for vid_id in annot_dict:
        split = annot_dict[vid_id]['subset']
        splitted_annot_dict[split][vid_id] = annot_dict[vid_id]
    annot_dict = splitted_annot_dict

    cnt_dict = {'training': -1, 'validation': -1}
    num_chunks = {'training': 64, 'validation': 8}
    n_all_vids = sum([len(annot_dict[split]) for split in annot_dict])
    cnt_vid = 0
    faulty = set()
    for split in annot_dict:

        #prepare per_split variables
        per_split_n_vids = len(annot_dict[split])
        per_chunk_n_vids = int(per_split_n_vids/num_chunks[split]) + 1
        per_split_cnt_vid = 0
        per_split_cnt_chunk = 0
        base_dir_path = os.path.join(out_path, split)
        os.makedirs(base_dir_path, exist_ok=True)
        
        cap_dict = {}
        hf = None
        for vid_id in annot_dict[split]:
            per_split_cnt_vid += 1

            if (per_split_cnt_vid % per_chunk_n_vids == 0) or (hf is None):
                if hf is not None:
                    hf.close()
                hf = h5py.File(os.path.join(base_dir_path, 'data_part_{}.h5'.format(per_split_cnt_chunk)), 'w')
                per_split_cnt_chunk += 1

            cnt_vid += 1  
            recipe = annot_dict[split][vid_id]['recipe_type']
            annots = annot_dict[split][vid_id]['annotations']
            
            feat_path = os.path.join(in_path,split,recipe,vid_id+'.npy')
            if not check_exists(feat_path):
                faulty.add(vid_id)
                continue

            features = np.load(feat_path)

            seg_dict = {'{}'.format(n): {} for n in range(len(annots))}
            for annot in annots:
                cnt_dict[split]+=1
                seg_id = str(annot['id'])
                start = annot['segment'][0]
                end = annot['segment'][1]
                image_id = str(cnt_dict[split])
                caption = annot['sentence']

                seg_features = features[start:end,:]
                if avg_pool:
                    seg_features = np.average(seg_features, axis=(1,2))
                else:
                    seg_features = append_pos_ids(seg_features)
                    seg_features = np.reshape(seg_features, [-1, seg_features.shape[-1]])

                hf.create_dataset('/'.join([vid_id, seg_id, 'image_id']), data=image_id)
                hf.create_dataset('/'.join([vid_id, seg_id, 'caption']), data=caption)
                hf.create_dataset('/'.join([vid_id, seg_id, 'features']), data=seg_features)

                seg_dict[seg_id] = {'image_id': image_id,
                                    'caption': caption}

            cap_dict[vid_id] = seg_dict                     
            
            print('{}/{}'.format(cnt_vid, n_all_vids))

        hf.close()
        print('\nSplit "{}" done.\nGenerating COCO-format captions...'.format(split))
        coco_format_caps = generate_coco_format_captions(cap_dict)
        save_path = os.path.join(base_dir_path, 'coco_format_caps.json'.format(split))
        json.dump(coco_format_caps, open(save_path, 'w'))

        print('Resuming...\n\n')

    print('Storing finished with {} faulty features.'.format(len(faulty)))

def store_vtt(in_path, out_path, annot_path, avg_pool):
    annot_dict_path = os.path.join(annot_path, 'all_splits.json')
    annot_dict = json.load(open(annot_dict_path, 'r'))
    
    #split annotation dictionary based on data_split_type
    splitted_annot_dict = {'training': {}, 'validation': {}, 'testing': {}}
    for vid_id in annot_dict:
        split = annot_dict[vid_id]['split']
        splitted_annot_dict[split][vid_id] = annot_dict[vid_id]
    annot_dict = splitted_annot_dict

    cnt_cap_dict = {'training': -1, 'validation': -1, 'testing': -1}
    cnt_img_dict = {'training': -1, 'validation': -1, 'testing': -1}
    num_chunks = {'training': 32, 'validation': 4, 'testing': 16}
    n_all_vids = sum([len(annot_dict[split]) for split in annot_dict])
    cnt_vid = 0
    faulty = set()
    for split in annot_dict:

        #prepare per_split variables
        per_split_n_vids = len(annot_dict[split])
        per_chunk_n_vids = int(per_split_n_vids/num_chunks[split]) + 1
        per_split_cnt_vid = 0
        per_split_cnt_chunk = 0
        base_dir_path = os.path.join(out_path, split)
        os.makedirs(base_dir_path, exist_ok=True)
        
        cap_dict = {}
        hf = None
        for vid_id in annot_dict[split]:
            per_split_cnt_vid += 1

            if (per_split_cnt_vid % per_chunk_n_vids == 0) or (hf is None):
                if hf is not None:
                    hf.close()
                hf = h5py.File(os.path.join(base_dir_path, 'data_part_{}.h5'.format(per_split_cnt_chunk)), 'w')
                hf.attrs['multi-caption'] = True
                per_split_cnt_chunk += 1

            cnt_vid += 1  
            annots = annot_dict[split][vid_id]['annotations']
            
            feat_path = os.path.join(in_path,vid_id+'.npy')
            if not check_exists(feat_path):
                faulty.add(vid_id)
                continue

            features = np.load(feat_path)

            seg_dict = {'{}'.format(n): {} for n in range(len(annots))}
            for seg_id in annots:
                cnt_cap_dict[split]+=1
                caption_id = cnt_cap_dict[split]
                caption = annots[seg_id]

                seg_features = features
                if avg_pool:
                    seg_features = np.average(seg_features, axis=(1,2))
                else:
                    seg_features = append_pos_ids(seg_features)
                    seg_features = np.reshape(seg_features, [-1, seg_features.shape[-1]])

                hf.create_dataset('/'.join([vid_id, seg_id, 'caption']), data=caption)
                if seg_id == '0':
                    cnt_img_dict[split] += 1
                    image_id = str(cnt_img_dict[split])
                    hf.create_dataset('/'.join([vid_id, seg_id, 'features']), data=seg_features)
                    hf.create_dataset('/'.join([vid_id, seg_id, 'image_id']), data=image_id)
                    
                seg_dict[seg_id] = {'image_id': image_id,
                                    'caption': caption,
                                    'caption_id': caption_id}

            cap_dict[vid_id] = seg_dict                     
            
            print('{}/{}'.format(cnt_vid, n_all_vids))

        hf.close()
        print('\nSplit "{}" done.\nGenerating COCO-format captions...'.format(split))
        coco_format_caps = generate_coco_format_multi_ref_captions(cap_dict)
        save_path = os.path.join(base_dir_path, 'coco_format_caps.json'.format(split))
        json.dump(coco_format_caps, open(save_path, 'w'))

        print('Resuming...\n\n')

    print('Storing finished with {} faulty features.'.format(len(faulty)))

def main(dataset, feat_type, in_path, out_path, annot_path):
    avg_pool = (feat_type == 'avg_pool')
    if dataset == 'youcook2':
        store_youcook2(in_path, out_path, annot_path, avg_pool)
    elif dataset == 'activity_net':
        store_activity_net(in_path, out_path, annot_path, avg_pool)
    elif dataset == 'msr_vtt':
        store_vtt(in_path, out_path, annot_path, avg_pool)
    

if __name__ == '__main__':
    description = 'The main feature store call function.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', type=str, default='youcook2')
    parser.add_argument('--feat_type', type=str, default='avg_pool')
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--annot_path', type=str)

    args = parser.parse_args()
    dataset = args.dataset
    feat_type = args.feat_type
    in_path = args.in_path
    out_path = args.out_path
    annot_path = args.annot_path
    

    main(dataset, feat_type, in_path, out_path, annot_path)
import os, json, pickle
import h5py

class GetMappings():
    def __init__(self,
                dataset='youcook2',
                h5_path='',
                annots_path=''):
        assert dataset in ['youcook2', 'activity_net']
        assert os.path.exists(h5_path)
        assert os.path.exists(annots_path)
        self.dataset = dataset
        self._hdf5_paths = self.list_hdf5(h5_path,'.h5')
        self.annots = json.load(open(annots_path, 'r'))
        
    def get_img_id2vid_id(self, file):
        with h5py.File(file, 'r') as hf:
            multi_cap = 'multi-caption' in hf.attrs
            for vid_id in hf:
                if multi_cap:
                    image_id = hf[vid_id]['0']['image_id'][()]

                    yield image_id, vid_id, '0'

                else:
                    for seg_id in hf[vid_id]:
                        image_id = hf[vid_id][seg_id]['image_id'][()]

                        yield image_id, vid_id, seg_id

    def list_hdf5(self,data_path,extension):
        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if os.path.isfile(os.path.join(root, file)):
                    if extension in file:
                        all_files.append(os.path.join(root, file))

        all_files = sorted(all_files)
        return all_files
    
    def get_annots(self, vid_id, seg_id):
        start = None
        end = None
        caption = None
        
        if self.dataset == 'youcook2':
            annots = self.annots['database'][vid_id]['annotations']
            for annot in annots:
                seg_id_annot = str(annot['id'])
                if seg_id_annot == seg_id:
                    start = annot['segment'][0]
                    end = annot['segment'][1]
                    caption = annot['sentence']
                    break
                
        elif self.dataset == 'activity_net':
            annots = self.annots[vid_id]
            n_annots = len(annots['sentences'])
            timestamp = annots['timestamps'][int(seg_id)]
            caption = annots['sentences'][int(seg_id)]

            start = int(np.floor(timestamp[0]))
            end = int(np.ceil(timestamp[1]))

        return caption, start, end
    
    def _to_url(self, vid_id, start, end):
        if self.dataset == 'activity_net':
            vid_id = vid_id[2:]

        return 'https://www.youtube.com/embed/{}?start={}&end={}'.format(vid_id, start, end)

    def __call__(self):
        dict_ = {}
        for file in self._hdf5_paths:
            for image_id, vid_id, seg_id in self.get_img_id2vid_id(file):
                caption, start, end = self.get_annots(vid_id, seg_id)
                dict_[image_id] = {'seg_id': seg_id,
                                   'vid_id': vid_id,
                                   'yt_url': self._to_url(vid_id, start, end),
                                   'start': start,
                                   'end': end,
                                   'caption': caption}
        return dict_
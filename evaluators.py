import os, sys, yaml, time, h5py
from contextlib import contextmanager
import numpy as np
import pickle, json, cv2
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from datetime import timedelta
from dataflow import RNGDataFlow, BatchData
from vid_cap import VidCapModel
from transformers import T5Tokenizer
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from typing import Dict, Optional
from threading import Thread
import logging

logger = logging.getLogger(__name__)

@tf.function
def prepare_inputs_for_evaluation(inputs, task):
    (vid, 
     vid_attn_mask, 
     txt, 
     txt_attn_mask,
     ) = inputs
    
    if task == 'next_word_prediction':
        txt_inputs = txt[:, :-1]
        txt_inputs_attn_mask = txt_attn_mask[:, :-1]

        txt_labels = txt[:, 1:]
        txt_labels_attn_mask = txt_attn_mask[:, 1:]

    else:
        raise ValueError('Please pass a valid task.')

    inputs = {'vid_inputs': vid,
              'vid_inputs_attn_mask': vid_attn_mask,
              'txt_inputs': txt_inputs,
              'txt_inputs_attn_mask': txt_inputs_attn_mask,
              'training': False}

    labels = {'txt_labels': txt_labels,
              'txt_labels_attn_mask': txt_labels_attn_mask}

    return inputs, labels

def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_tsv: TSV file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    assert label_file.endswith('.json')
    if res_file.endswith('.tsv'):
        res_file_coco = op.splitext(res_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(res_file, res_file_coco)
    elif res_file.endswith('.json'):
        res_file_coco = res_file
    else:
        raise ValueError('unknown prediction result file format: {}'.format(res_file))
    
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result

class Generator():
    def __init__(self,
                 data_path,
                 max_vid_length,
                 max_txt_length):
        Generator.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        Generator.filenames = Generator.list_hdf5(data_path, '.h5')
        Generator.max_vid_length = max_vid_length
        Generator.max_txt_length = max_txt_length
        Generator.bos = Generator.tokenizer.pad_token
        Generator.eos = Generator.tokenizer.eos_token
        self._calc_stats()

    def _calc_stats(self):
        self.num_samples = 0
        for hf in [h5py.File(file, 'r') for file in Generator.filenames]:
            multi_cap = 'multi-caption' in hf.attrs
            for vid_id in hf:
                if multi_cap:
                    self.num_samples += 1
                else:
                    self.num_samples += len(hf[vid_id].keys())

    @staticmethod
    def list_hdf5(data_path,extension):
        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if os.path.isfile(os.path.join(root, file)):
                    if extension in file:
                        all_files.append(os.path.join(root, file))
        
        all_files = sorted(all_files)
        return all_files
    
    @staticmethod
    def pad_data(vid_features, caption):
        # trimming video features
        vid_features = vid_features[:Generator.max_vid_length,:]
        vid_attention_mask = np.ones((vid_features.shape[0],))
        vid_outputs = [vid_features, vid_attention_mask]

        (vid_features, 
         vid_attention_mask, 
         ) = [Generator.check_pad(out,Generator.max_vid_length,0,'constant') for out in vid_outputs]

        # tokenizing and trimming caption
        caption = Generator.bos + caption + Generator.eos
        caption_ids = Generator.tokenizer.encode(caption)[:Generator.max_txt_length]
        caption_ids = np.array(caption_ids).astype('int32')
        cap_attention_mask = np.ones((caption_ids.shape[0],))
        cap_outputs = [caption_ids, cap_attention_mask]

        (caption_ids, 
         cap_attention_mask, 
         ) = [Generator.check_pad(out,Generator.max_txt_length,0,'constant') for out in cap_outputs]

        return (vid_features, 
                vid_attention_mask, 
                caption_ids, 
                cap_attention_mask,
                )

    @staticmethod
    def check_pad(inputs,step,axis,mode='edge'):
        inp_shp = inputs.shape[axis]
        diff = step-inp_shp
        if diff==0:
            return inputs
        elif diff==step:
            shp = list(inputs.shape)
            shp[axis] = step
            return np.zeros(shp)
        else:   
            pad_width = []
            for n in range(len(inputs.shape)):
                if n==axis:
                    pad_width.append((0,diff))
                else:
                    pad_width.append((0,0))
            return np.pad(inputs,pad_width,mode=mode)

    def __len__(self):
        return self.num_samples

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            multi_cap = 'multi-caption' in hf.attrs
            for vid_id in hf:
                if multi_cap:
                    vid_features = hf[vid_id]['0']['features'][()]
                    image_id = hf[vid_id]['0']['image_id'][()]
                    
                    #only taking first caption for now
                    caption = hf[vid_id]['0']['caption'][()]
                    
                    (vid_features, 
                     vid_attention_mask, 
                     caption_ids, 
                     cap_attention_mask,
                    ) = Generator.pad_data(vid_features, caption)

                    yield vid_features, vid_attention_mask, image_id, caption_ids, cap_attention_mask
                
                else:
                    for seg_id in hf[vid_id]:
                        vid_features = hf[vid_id][seg_id]['features'][()]
                        image_id = hf[vid_id][seg_id]['image_id'][()]
                        caption = hf[vid_id][seg_id]['caption'][()]

                        (vid_features, 
                         vid_attention_mask, 
                         caption_ids, 
                         cap_attention_mask,
                        ) = Generator.pad_data(vid_features, caption)
                        
                        yield vid_features, vid_attention_mask, image_id, caption_ids, cap_attention_mask

class Evaluator():
    def __init__(self,
                 config,
                 strategy=None,
                 **kwargs):

        print('Initializing evaluator...')
        self.config = config

        # get test/data parameters
        self.run_name = self.config['TRAIN']['RUN_NAME']
        self.outputs_path = self.config['TRAIN']['OUTPUTS_PATH']
        self.ckpt_save_period = self.config['TRAIN']['SNAPSHOTS']['CKPT_SAVE_PERIOD']
        self.labels_path = self.config['DATA']['COCO_LABELS_PATH']
        self.feat_path = self.config['DATA']['TEST_FEATURES_PATH']
        self.max_vid_length = self.config['DATA']['MAX_FRAMES'] * self.config['DATA']['MAX_REGIONS']
        self.max_txt_length = self.config['DATA']['MAX_TXT_LENGTH']
        self.block_length = self.config['DATA']['BLOCK_LENGTH']
        self.d_vid = self.config['VID_CONFIG']['D_VID']
        self.enable_gpu = VidCapModel._check_generation_graph_capability(self.config['TX_CONFIG']['GENERATION'])
        self.coco_metrics = []
        self.ckpt_path = kwargs.get('ckpt_path', None)

        tf.get_logger().setLevel('ERROR')
        tf.debugging.set_log_device_placement(False)

        if strategy is None and self.enable_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
            self.num_repl = self.strategy.num_replicas_in_sync
            print('Successfully allocated {} workers for evaluation.'.format(self.num_repl))
        elif strategy is None and not self.enable_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
        else:
            self.strategy = strategy
        
        self.batch_size = self.config['TEST']['BATCH_SIZE']
        
        if self.enable_gpu:
            scope = self.strategy.scope()
        else:
            scope = tf.device('cpu:0')

        with scope:
            self.df = Generator(data_path=self.feat_path,
                                max_vid_length=self.max_vid_length,
                                max_txt_length=self.max_txt_length)

            self.dataset = tf.data.Dataset.from_tensor_slices(self.df.filenames)
            self.dataset = self.dataset.interleave(lambda filename: tf.data.Dataset.from_generator(
                                                self.df, 
                                                (tf.float32, 
                                                 tf.int32,
                                                 tf.string,
                                                 tf.int32,
                                                 tf.int32),
                                                (tf.TensorShape([self.max_vid_length, self.d_vid]), 
                                                 tf.TensorShape([self.max_vid_length,]),
                                                 tf.TensorShape(()),
                                                 tf.TensorShape([self.max_txt_length,]),
                                                 tf.TensorShape([self.max_txt_length,])),
                                                args=(filename,)),
                                                cycle_length=tf.data.experimental.AUTOTUNE, 
                                                block_length=self.block_length,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                                deterministic=True)

            self.dataset = (self.dataset.cache()
                                        .batch(self.batch_size)
                                        .prefetch(tf.data.experimental.AUTOTUNE)
                            )
            
            if self.enable_gpu:
                self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)

            self.iter_dataset = iter(self.dataset)
        self.steps_max = int( np.ceil(len(self.df) / self.batch_size) )
        self.metrics = {}
        self.logs_path = os.path.join(self.outputs_path,'eval_logs',self.run_name)
        self.coco_numbers_path = os.path.join(self.outputs_path,'coco_numbers',self.run_name)
        self.inter_states_path = os.path.join(self.outputs_path,'intermediate_states',self.run_name)
        self.writer = tf.summary.create_file_writer(logdir=self.logs_path)
        tf.summary.trace_on(graph=False, profiler=False)
        os.makedirs(self.coco_numbers_path, exist_ok=True)
        os.makedirs(self.inter_states_path, exist_ok=True)
        self._init_metrics()

    def add_metric(self,
                   metric):
        names = metric.name.split('/')
        if len(names)!=2:
            raise ValueError('Please pass a metric with name pattern: matric_type/metric_name')
        mtype, mname = names
        if mtype not in self.metrics:
            self.metrics[mtype] = {}
        self.metrics[mtype].update({mname: metric})

    def reset_metrics(self):
        for mtype in self.metrics:
            for mname in self.metrics[mtype]:
                self.metrics[mtype][mname].reset_states()

    def update_tf_summary(self, step):
        with self.writer.as_default():
            for mtype in self.metrics:
                for mname in self.metrics[mtype]:
                    summary_name = '{}/{}'.format(mtype, mname)
                    summary_result = self.metrics[mtype][mname].result().numpy()
                    if mtype == 'Losses' or mtype == 'Accuracies' or mtype == 'COCO Numbers':
                            tf.summary.scalar(summary_name, summary_result, step=step)
                    elif mtype == 'Distributions':
                            tf.summary.histogram(summary_name, summary_result, step=step)
                            self.metrics[mtype][mname].reset_states()
    
    def save_coco_top_k(self, results, k, metric):
        metric_results = []
        for result in results:
            metric_results.append(result[metric])

        top_k_idx = np.argsort(metric_results)[-k:][::-1]
        top_k_results = {'{}'.format((n+1)*self.ckpt_save_period): results[n] for n in top_k_idx}
        open(os.path.join(self.coco_numbers_path,'top_{}_results_{}.json'.format(k,metric)), 'w').write(json.dumps(top_k_results, indent=2))

        return results[top_k_idx[0]][metric]

    def update_coco_metrics(self, metrics):
        for mname in metrics:
            self.metrics['COCO Numbers'][mname].update_state(metrics[mname])

    def pretty_progress(self,
                        step,
                        steps_max,
                        t_step,
                        **metrics):

        eta_secs = min(2**32, int((steps_max - step) * t_step))

        if not hasattr(self, 'ema_eta'):
            self.ema_eta = 0
            self.ema_alpha = 0.1
        
        self.ema_eta = int(self.ema_alpha * eta_secs + (1-self.ema_alpha) * self.ema_eta)
        
        progress_str = ''
        progress_str += 'Iter {}/{}'.format(step+1,steps_max)
        for metric in metrics:
            progress_str += ' - {}: {:0.4f}'.format(metric, metrics[metric])
        
        progress_str += '{:>5} //{:>5} ETA {:0>8}, {:0.2f}/step {:>10}'.format('', '', str(timedelta(seconds=self.ema_eta)), t_step, '')
        progress_str += '\n'
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    def _compute_loss_0(self,
                        labels,
                        predictions,
                        mask=None,
                        reduction=tf.losses.Reduction.AUTO):
        loss_object = tf.keras.losses.SparseCategoricalAccuracy(reduction = reduction)
        loss_ = loss_object(labels, predictions)
        return loss_

    def _init_metrics(self):
        self.add_metric(Mean(name='COCO Numbers/Bleu_1'))
        self.add_metric(Mean(name='COCO Numbers/Bleu_2'))
        self.add_metric(Mean(name='COCO Numbers/Bleu_3'))
        self.add_metric(Mean(name='COCO Numbers/Bleu_4'))
        self.add_metric(Mean(name='COCO Numbers/METEOR'))
        self.add_metric(Mean(name='COCO Numbers/ROUGE_L'))
        self.add_metric(Mean(name='COCO Numbers/CIDEr'))
        self.add_metric(Mean(name='COCO Numbers/SPICE'))

    def _compute_loss(self, labels, predictions, mask=None):
        per_example_loss = self._compute_loss_0(labels = labels,
                                                predictions = predictions,
                                                reduction = tf.keras.losses.Reduction.NONE)

        avg_loss = tf.reduce_mean(per_example_loss) #(1/n)*L ; n is size of miniBatch
        
        self.metrics['Losses']['Test_Loss'].update_state(avg_loss)
        self.metrics['Accuracies']['Test_Accuracy'].update_state(labels, predictions)

        return avg_loss

    def _decoded_to_coco_numbers(self, eval_step, coco_format_results, from_ids):
        if from_ids:
            for image_id in coco_format_results:
                coco_format_results[image_id]['caption'] = self.model.tokenizer.decode(coco_format_results[image_id]['caption'], 
                                                                                       skip_special_tokens=True)

        print('Writing results in COCO format...')
        #sorting by image_ids
        sorted_ids = sorted([int(key) for key in coco_format_results.keys()])
        coco_format_results = [coco_format_results[str(key)] for key in sorted_ids]
        json.dump(coco_format_results, open(os.path.join(self.coco_numbers_path,'captions_{}.json'.format(eval_step)), 'w'))
        print('Calculating COCO metrics...')
        metrics = evaluate_on_coco_caption(res_file=os.path.join(self.coco_numbers_path,'captions_{}.json'.format(eval_step)), 
                                           label_file=self.labels_path)
        open(os.path.join(self.coco_numbers_path,'results_{}.json'.format(eval_step)), 'w').write(json.dumps(metrics, indent=2))
        self.update_coco_metrics(metrics)
        self.update_tf_summary(step = eval_step)
        self.coco_metrics.append(metrics)
        _ = self.save_coco_top_k(self.coco_metrics, k=5, metric='Bleu_4')
        top_CIDEr = self.save_coco_top_k(self.coco_metrics, k=5, metric='CIDEr')

        logger.info("EVALERR: {}%".format(round(100 * top_CIDEr, 4)))
        
        print('\nEvaluation for model at step {} finished.'.format(eval_step))

    def _save_intermediate_states(self, eval_step, outputs_all):
        print('Saving intermediate states...')
        pickle.dump(outputs_all,
                    open(os.path.join(self.inter_states_path,'states_{}.pkl'.format(eval_step)), 'wb'),
                    protocol=4)
        
        print('\nSaving intermediate states for model at step {} finished.'.format(eval_step))

    def _get_ckpt_paths(self, ckpts_dir):
        prefix = 'ckpt-'
        all_ckpts = []
        for root, dirs, files in os.walk(ckpts_dir):
            for file in files:
                if os.path.isfile(os.path.join(root, file)):
                    if prefix in file:
                        file_path = os.path.join(root, file)
                        all_ckpts.append(os.path.splitext(file_path)[0])
        all_ckpts = set(all_ckpts)
        all_ckpts = {'{}'.format(ckpt_path.split('ckpt-')[-1]): ckpt_path for ckpt_path in all_ckpts}
        all_ckpts = {str(eval_step): all_ckpts[str(eval_step)] for eval_step in sorted([int(key) for key in all_ckpts.keys()])}
        ckpt_save_period = int(list(all_ckpts.keys())[1])-int(list(all_ckpts.keys())[0])

        return all_ckpts, ckpt_save_period

    def test_gpu(self, eval_step=0, refresh_ckpt_path=None):
        @tf.function(input_signature=[self.iter_dataset.element_spec])
        def _test_step_dist(dist_inputs):
            def _step_fn(inputs):
                vid_inputs, vid_mask, image_ids, _ , _ = inputs

                decoded_ids = self.model.generate(vid_inputs=vid_inputs,
                                                  vid_mask=vid_mask)


                return decoded_ids, image_ids

            decoded_ids_batch, image_ids_batch = self.strategy.run(_step_fn, args=(dist_inputs,))
            
            per_replica_decoded_ids = self.strategy.experimental_local_results(decoded_ids_batch)
            per_replica_image_ids = self.strategy.experimental_local_results(image_ids_batch)
            
            #decoded_ids = tf.concat(per_replica_decoded_ids, axis=0)
            #image_ids = tf.concat(per_replica_image_ids, axis=0)
            
            #return decoded_ids, image_ids
            return per_replica_decoded_ids, per_replica_image_ids

        coco_format_ids = {}
        with self.strategy.scope():
            if refresh_ckpt_path is not None:
                #refresh model weights based on ckpt_path
                #special use: loop over a directory of checkpoints
                self.checkpoint.restore(refresh_ckpt_path).assert_consumed()

            print('Evaluating model at step {} ...'.format(eval_step))
            coco_format_results = {}
            self.reset_metrics()
            self.iter_dataset = iter(self.dataset)
            for step in range(self.steps_max):
                
                t_step = time.time()
                inputs = self.iter_dataset.get_next()
                per_n_decoded_ids, per_n_image_ids = _test_step_dist(inputs)
                
                for n in range(len(per_n_decoded_ids)):
                    decoded_ids = per_n_decoded_ids[n].numpy()
                    image_ids = per_n_image_ids[n].numpy()
                    for decoded_id, image_id in zip(decoded_ids, image_ids):
                        image_id = image_id.decode()
                        coco_format_ids[image_id]={'image_id': image_id,
                                                   'caption': decoded_id}
                
                t_step = time.time() - t_step
                
                self.pretty_progress(step=step,
                                     steps_max=self.steps_max,
                                     t_step=t_step)

                
        Thread(target=self._decoded_to_coco_numbers, args=(eval_step, coco_format_ids, True)).start()

    def test_cpu(self, eval_step=0, refresh_ckpt_path=None, silent=False):
        if refresh_ckpt_path is not None:
            #refresh model weights based on ckpt_path
            #special use 1: multiple evaluation within train loop
            #special use 2: loop over a directory of checkpoints
            self.checkpoint.restore(refresh_ckpt_path).assert_consumed()

        def _test_step(inputs):
            vid_inputs, vid_mask, image_ids, _ , _ = inputs
            decoded_ids = self.model.generate(vid_inputs=vid_inputs,
                                              vid_mask=vid_mask)

            captions = self.model.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
            image_ids = [img_id.decode() for img_id in image_ids.numpy()]
            return captions, image_ids

        with tf.device('cpu:0'):
            print('Evaluating model at step {} ...'.format(eval_step))
            coco_format_results = {}
            self.reset_metrics()
            self.iter_dataset = iter(self.dataset)
            for step in range(self.steps_max):
                
                t_step = time.time()
                inputs = self.iter_dataset.get_next()
                captions, image_ids = _test_step(inputs)
                t_step = time.time() - t_step
                
                self.pretty_progress(step=step,
                                     steps_max=self.steps_max,
                                     t_step=t_step,
                                     )

                for caption, image_id in zip(captions, image_ids):
                    coco_format_results[image_id]={'image_id': image_id,
                                                   'caption': caption}
        
        self._decoded_to_coco_numbers(eval_step, coco_format_results, False)

    def intermediate_gpu(self, eval_step=0, refresh_ckpt_path=None):
        @tf.function(input_signature=[self.iter_dataset.element_spec])
        def _test_step_dist(dist_inputs):
            def _step_fn(inputs):
                vid_features, vid_attention_mask, image_ids, caption_ids, cap_attention_mask = inputs
                inputs_nwp, _ = prepare_inputs_for_evaluation((vid_features, 
                                                               vid_attention_mask,
                                                               caption_ids,
                                                               cap_attention_mask), task='next_word_prediction')
                outputs_nwp = self.model(**inputs_nwp)
                return outputs_nwp, image_ids

            def _get_vq_states(vq_states):
                agg_embds = ()
                agg_scores = ()
                agg_ids = ()
                for layer_state in vq_states:
                    if layer_state != {}:
                        local_layer_embd = self.strategy.experimental_local_results(layer_state['output_embeddings'])
                        local_layer_score = self.strategy.experimental_local_results(layer_state['output_scores'])
                        local_layer_id = self.strategy.experimental_local_results(layer_state['output_ids'])
                        agg_layer_embd = tf.concat(local_layer_embd, axis=0)
                        agg_layer_score = tf.concat(local_layer_score, axis=0)
                        agg_layer_id = tf.concat(local_layer_id, axis=0)
                        agg_embds = agg_embds + (agg_layer_embd,)
                        agg_scores = agg_scores + (agg_layer_score,)
                        agg_ids = agg_ids + (agg_layer_id,)
                return {'embeddings': agg_embds, 'scores': agg_scores, 'ids': agg_ids}

            def _get_attn_weights(attn_weights):
                agg_attn_weights = ()
                for layer_attn in attn_weights:
                    local_layer_attn = self.strategy.experimental_local_results(layer_attn)
                    agg_layer_attn = tf.concat(local_layer_attn, axis=0)
                    agg_attn_weights = agg_attn_weights + (agg_layer_attn,)
                return agg_attn_weights

            outputs_batch, image_ids_batch = self.strategy.run(_step_fn, args=(dist_inputs,))
            
            encoder_self_vq_states = outputs_batch['encoder_outputs']['attention_vq_states']
            encoder_self_attn_weights = outputs_batch['encoder_outputs']['attention_weights']
            decoder_self_vq_states = outputs_batch['decoder_outputs']['self_attention_vq_states']
            decoder_self_attn_weights = outputs_batch['decoder_outputs']['self_attention_weights']
            decoder_cross_vq_states = outputs_batch['decoder_outputs']['cross_attention_vq_states']
            decoder_cross_attn_weights = outputs_batch['decoder_outputs']['cross_attention_weights']
            decoder_predictions = self.strategy.experimental_local_results(outputs_batch['predictions'])
            decoder_predictions = tf.math.argmax(tf.concat(decoder_predictions, axis=0), axis=-1)

            outputs = {'encoder_attention_weights': _get_attn_weights(encoder_self_attn_weights),
                       'decoder_self_attention_weights': _get_attn_weights(decoder_self_attn_weights),
                       'decoder_cross_attention_weights': _get_attn_weights(decoder_cross_attn_weights),
                       'encoder_vq_states': _get_vq_states(encoder_self_vq_states),
                       'decoder_self_vq_states': _get_vq_states(decoder_self_vq_states),
                       'decoder_cross_vq_states': _get_vq_states(decoder_cross_vq_states),
                       'decoder_predictions': decoder_predictions}

            per_replica_image_ids = self.strategy.experimental_local_results(image_ids_batch)
            image_ids = tf.concat(per_replica_image_ids, axis=0)
            
            return outputs, image_ids

        outputs_all = {}
        with self.strategy.scope():
            def convert_state_to_np(states):
                n_layers = len(states)
                agg_states = None
                for n in range(n_layers):
                    layer_state = states[n].numpy()
                    if agg_states is None:
                        state_shape = layer_state.shape
                        agg_states = np.zeros((n_layers,)+state_shape, dtype='float32')
                    agg_states[n,:] = layer_state
                return agg_states

            if refresh_ckpt_path is not None:
                #refresh model weights based on ckpt_path
                #special use: loop over a directory of checkpoints
                self.checkpoint.restore(refresh_ckpt_path).assert_consumed()

            print('Evaluating model at step {} ...'.format(eval_step))
            self.reset_metrics()
            self.iter_dataset = iter(self.dataset)
            for step in range(self.steps_max):
                t_step = time.time()
                inputs = self.iter_dataset.get_next()
                outputs, image_ids = _test_step_dist(inputs)
                image_ids = image_ids.numpy()
                for key in outputs:
                    if 'attention_weights' in key:
                        outputs[key] = convert_state_to_np(outputs[key])
                    elif 'vq_states' in key:
                        outputs[key] = {'embeddings': convert_state_to_np(outputs[key]['embeddings']),
                                        'scores': convert_state_to_np(outputs[key]['scores']),
                                        'ids': convert_state_to_np(outputs[key]['ids'])}
                    elif 'predictions' in key:
                        outputs[key] = outputs[key].numpy()

                for n, image_id in enumerate(image_ids):
                    image_id = image_id.decode()
                    outputs_all[image_id]={'image_id': image_id,
                                           'outputs': {}}
                    for key in outputs:
                        if 'attention_weights' in key:
                            outputs_all[image_id]['outputs'][key] = outputs[key][:,n,:]
                        elif 'vq_states' in key:
                            outputs_all[image_id]['outputs'][key] = {'embeddings': outputs[key]['embeddings'][:,n,:],
                                                                     'scores': outputs[key]['scores'][:,n,:],
                                                                     'ids': outputs[key]['ids'][:,n,:]}
                        elif 'predictions' in key:
                            outputs_all[image_id]['outputs'][key] = outputs[key][n,:]

                t_step = time.time() - t_step
                
                self.pretty_progress(step=step,
                                     steps_max=self.steps_max,
                                     t_step=t_step)

                
        Thread(target=self._save_intermediate_states, args=(eval_step, outputs_all)).start()

    def test(self, eval_step=0, ckpts_dir=None, save_intermediate=False):
        loop_over_ckpts = False
        if ckpts_dir is not None:
            assert os.path.exists(ckpts_dir), "Please provide a valid checkpoint path."
            # we assume that checkpoints have been saved with the following format every 
            # ckpt_save_period steps: '/path/to/checkpoints/ckpt-eval_step.xxx'
            all_ckpts, self.ckpt_save_period = self._get_ckpt_paths(ckpts_dir)
            assert len(all_ckpts)>0, "No checkpoints found under the provided path."
            print('{} checkpoints found under provided path.\nBeginning evaluation...'.format(len(all_ckpts)))
            loop_over_ckpts = True

        if self.enable_gpu:
            if save_intermediate:
                eval_fn = self.intermediate_gpu
            else:
                eval_fn = self.test_gpu
        else:
            eval_fn = self.test_cpu

        if loop_over_ckpts:
            for n, eval_step in enumerate(all_ckpts):
                print('====Evaluation {}/{}'.format(n,len(all_ckpts)))
                eval_fn(eval_step=int(eval_step), refresh_ckpt_path=all_ckpts[eval_step])
        else:
            eval_fn(eval_step)
            
class VID_CAP_Evaluator(Evaluator):
    def __init__(self,
                 config,
                 **kwargs):
        super(VID_CAP_Evaluator, self).__init__(config=config, **kwargs)

        if self.enable_gpu:
            scope = self.strategy.scope()
        else:
            scope = tf.device('cpu:0')

        with scope:
            print('Building video model for evaluation...')
            self.model = VidCapModel(config=self.config)
            self.checkpoint = tf.train.Checkpoint(model=self.model)

            print('Initializing evaluation variables...')
            if self.enable_gpu:
                self.model.set_strategy(self.strategy)
                self.model.distributed_init(batch_size=self.batch_size, d_vid=self.d_vid)
            else:
                self.model.init(batch_size=self.batch_size, d_vid=self.d_vid)

            if self.ckpt_path is not None:
                print('Loading variables from checkpoint...')
                self.checkpoint.restore(self.ckpt_path).assert_consumed()

            print('All evaluation initializations done.')
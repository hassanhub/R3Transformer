import os, sys, yaml, time, random
from queue import Queue
from threading import Thread, Lock
import numpy as np
import nvidia_smi
import pickle, json, cv2, h5py, re
import tensorflow as tf
from tensorflow.keras.metrics import Metric, Mean, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD
from datetime import timedelta
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from dataflow import RNGDataFlow, MultiProcessRunnerZMQ, BatchData, MultiThreadMapData, MultiProcessMapDataZMQ
from vid_cap import VidCapModel
from transformers import T5Tokenizer
from evaluators import Evaluator, VID_CAP_Evaluator
import logging

logger = logging.getLogger(__name__)

@tf.function
def prepare_inputs_for_training(inputs, task):
    (vid, 
     vid_attn_mask, 
     txt, 
     txt_attn_mask, 
     txt_masked, 
     txt_mlm_mask,
     ) = inputs
    
    if task == 'next_word_prediction':
        txt_inputs = txt[:, :-1]
        txt_inputs_attn_mask = txt_attn_mask[:, :-1]

        txt_labels = txt[:, 1:]
        txt_labels_attn_mask = txt_attn_mask[:, 1:]

    elif task == 'masked_language_modeling':
        txt_inputs = txt_masked
        txt_inputs_attn_mask = txt_attn_mask[:, tf.newaxis, :]

        txt_labels = txt
        txt_labels_attn_mask = txt_mlm_mask

    elif task == 'causal_masked_language_modeling':
        txt_inputs = txt_masked[:, :-1]
        txt_inputs_attn_mask = txt_attn_mask[:, :-1]

        batch_size, mask_seq_length = get_shape(txt_inputs_attn_mask)
        seq_ids = tf.range(mask_seq_length)
        causal_mask = tf.less_equal(
            tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None])
        causal_mask = tf.cast(causal_mask, dtype=tf.int32) # (batch_size, from_seq_len, to_seq_len)
        
        txt_inputs_attn_mask = causal_mask * txt_inputs_attn_mask[:, tf.newaxis, :] # (batch_size, from_seq_len, to_seq_len)

        txt_labels = txt[:, 1:]
        txt_labels_attn_mask = txt_attn_mask[:, 1:] 
    else:
        raise ValueError('Please pass a valid task.')

    inputs = {'vid_inputs': vid,
              'vid_inputs_attn_mask': vid_attn_mask,
              'txt_inputs': txt_inputs,
              'txt_inputs_attn_mask': txt_inputs_attn_mask,
              'training': True}

    labels = {'txt_labels': txt_labels,
              'txt_labels_attn_mask': txt_labels_attn_mask}

    return inputs, labels

def get_shape(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class Scalar(Metric):
    def __init__(self,
                 name='Scalar_Metric',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.scalar = self.add_weight(name='scalar_metric', dtype=tf.float32, initializer='zeros')

    def update_state(self, scalar):
        self.scalar.assign(scalar)

    def result(self):
        return self.scalar

class MovingAverage(Metric):
    def __init__(self,
                 alpha=0.01,
                 name='Moving_Average_Metric',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.ema = self.add_weight(name='moving_average_metric', dtype=tf.float32, initializer='zeros')

    def update_state(self, x):
        new_ema = self.alpha * x + (1-self.alpha) * self.ema
        self.ema.assign(new_ema)

    def result(self):
        return self.ema

    def reset_states(self):
        self.ema.assign(0.)

class SampleEvalThread(Thread): 
    def __init__(self, threadID, queue, lock, eval_fn): 
        super().__init__() 
        self.threadID = threadID 
        self.queue = queue
        self.lock = lock
        self.eval_fn = eval_fn

    @staticmethod        
    def process_data(evalQ, queueL, eval_fn): 
        while not Trainer.Eval_Exit_Flag: 
            queueL.acquire() 
            if not evalQ.empty(): 
                data = evalQ.get() 
                queueL.release()
                data['refresh_ckpt_path'] = data['refresh_ckpt_path'].join()
                with tf.device('cpu:0'):
                    eval_fn(**data)
            else: 
                queueL.release() 
                time.sleep(1)

    def run(self):
        SampleEvalThread.process_data(self.queue, self.lock, self.eval_fn)

class CkptThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class Generator():
    def __init__(self,
                 data_path,
                 max_vid_length,
                 max_txt_length):

        Generator.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        Generator.filenames = Generator.list_hdf5(data_path, '.h5')
        Generator.max_vid_length = max_vid_length
        Generator.max_txt_length = max_txt_length
        Generator.mask_rate = 0.2
        Generator.bos = Generator.tokenizer.pad_token
        Generator.eos = Generator.tokenizer.eos_token

        if Generator.tokenizer.mask_token is None:
            print('Setting mask_token: <mask>')
            Generator.tokenizer.add_special_tokens({'mask_token': '<mask>'})

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
        
        # tokenizing and trimming caption
        caption = Generator.bos + caption + Generator.eos
        caption_ids = Generator.tokenizer.encode(caption)[:Generator.max_txt_length]
        caption_ids = np.array(caption_ids).astype('int32')

        vid_attention_mask = np.ones((vid_features.shape[0],))
        cap_attention_mask = np.ones((caption_ids.shape[0],))

        vid_outputs = [vid_features, vid_attention_mask]
        cap_outputs = [caption_ids, cap_attention_mask]

        (vid_features, 
         vid_attention_mask, 
         ) = [Generator.check_pad(out,Generator.max_vid_length,0,'constant') for out in vid_outputs]

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
    
    @staticmethod
    @tf.function
    def tf_random_mask(vid_features, vid_attenion_mask, caption_ids, cap_attention_mask):
        seq_len = len(caption_ids)
        mask_ids = [0] * seq_len
        random_ids = [0] * seq_len
        no_touch_ids = [0] * seq_len
        randomness = tf.random.uniform((seq_len,3))
        random_vocabs = tf.random.uniform((seq_len,), maxval=Generator.tokenizer.vocab_size, dtype=tf.int32)
        for n in range(seq_len):
            if randomness[n,0] <= Generator.mask_rate:
                #do masking
                if randomness[n,1] <= 0.8:
                    # 80% mask
                    mask_ids[n] = Generator.tokenizer.mask_token_id

                elif randomness[n,2] <= 0.5:
                    # 10% replace with random token from vocab
                    random_ids[n] = random_vocabs[n]
                else:
                    # 10% do nothing but keep track of it
                    no_touch_ids[n] = 1
        masks = (tf.stack(mask_ids) + tf.stack(random_ids)) * cap_attention_mask
        mask_pos = tf.cast(tf.cast(masks, tf.bool), tf.int32)
        masked_caption_ids = caption_ids * (1-mask_pos) + masks
        mask_pos = tf.stack(no_touch_ids) * cap_attention_mask + mask_pos

        return vid_features, vid_attenion_mask, caption_ids, cap_attention_mask, masked_caption_ids, mask_pos

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            multi_cap = 'multi-caption' in hf.attrs
            for vid_id in hf:
                if multi_cap:
                    vid_features = hf[vid_id]['0']['features'][()]
                for seg_id in hf[vid_id]:
                    if not multi_cap:
                        vid_features = hf[vid_id][seg_id]['features'][()]
                    caption = hf[vid_id][seg_id]['caption'][()]
                    
                    (vid_features, 
                     vid_attention_mask, 
                     caption_ids, 
                     cap_attention_mask,
                    ) = Generator.pad_data(vid_features, caption)
                    
                    yield vid_features, vid_attention_mask, caption_ids, cap_attention_mask

class LRSchedule(LearningRateSchedule):
    def __init__(self,
                 max_steps,
                 max_warmup_steps,
                 lr_0,
                 lr_0_warmup,
                 anneal_method='linear',
                 **kwargs):
        super(LRSchedule, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_warmup_steps = max_warmup_steps
        self.lr_0 = lr_0
        self.lr_0_warmup = lr_0_warmup
        self.lr_max_warmup = lr_0
        self.last_lr = 0
        self.anneal_method = anneal_method
        self.minimum_lr = tf.constant(1e-7, dtype=tf.float32)

    def lr_cosine(self,step):
        return self.lr_0 * (tf.math.cos(np.pi * (step - self.max_warmup_steps) / self.max_steps) + 1.0) * 0.5

    def lr_linear(self,step):
        alpha = - self.lr_0 / self.max_steps
        lr = (step - self.max_warmup_steps) * alpha + self.lr_0
        return lr

    def lr_warmup(self,step):
        alpha = (self.lr_max_warmup - self.lr_0_warmup) / self.max_warmup_steps
        lr = step * alpha + self.lr_0_warmup
        return lr
    
    def lr_anneal(self,step):
        if self.anneal_method == 'cosine':
            return tf.maximum(self.lr_cosine(step), self.minimum_lr)
        elif self.anneal_method == 'linear':
            return tf.maximum(self.lr_linear(step), self.minimum_lr)

    def __call__(self,step):
        if self.max_warmup_steps == 0:
            self.last_lr = self.lr_anneal(step)
        else:
            self.last_lr = tf.minimum(self.lr_warmup(step), self.lr_anneal(step))
        return self.last_lr

class Trainer():
    def __init__(self,
                 config,
                 **kwargs):

        print('Initializing trainer...')
        self.config = config

        # get train/data parameters
        self.run_name = self.config['TRAIN']['RUN_NAME']
        self.base_lr_0 = self.config['TRAIN']['SOLVER']['BASE_LR_0']
        self.base_lr_0_warmup = self.config['TRAIN']['SOLVER']['BASE_LR_0_WARMUP']
        self.warmup_ratio = self.config['TRAIN']['SOLVER']['WARMUP_RATIO']
        self.batch_size = self.config['TRAIN']['BATCH_SIZE']
        self.max_steps = self.config['TRAIN']['MAX_STEPS']
        self.cross_device_ops = getattr(tf.distribute, self.config['TRAIN']['DISTRIBUTED']['CROSS_DEV_OPS'])
        self.num_packs = self.config['TRAIN']['DISTRIBUTED']['NUM_PACKS']
        self.seed = self.config['TRAIN']['SEED']
        self.pretrain_ckpt_path = self.config['TRAIN']['PRETRAIN_CKPT_PATH']
        self.outputs_path = self.config['TRAIN']['OUTPUTS_PATH']
        self.pre_train = self.pretrain_ckpt_path is not None
        self.lr_scale = self.config['TRAIN']['SOLVER']['LR_SCALE']
        self.lr_anneal = self.config['TRAIN']['SOLVER']['LR_ANNEAL']
        self.lr_r3 = self.config['TRAIN']['SOLVER']['LR_R3']
        self.lr_0 = self.lr_scale * self.base_lr_0
        self.lr_0_warmup = self.lr_scale * self.base_lr_0_warmup
        self.train_eval_period = self.config['TRAIN']['SNAPSHOTS']['TRAIN_EVAL_PERIOD']
        self.tb_update_period = self.config['TRAIN']['SNAPSHOTS']['TB_UPDATE_PERIOD']
        self.ckpt_save_period = self.config['TRAIN']['SNAPSHOTS']['CKPT_SAVE_PERIOD']
        self.feat_path = self.config['DATA']['TRAIN_FEATURES_PATH']
        self.max_vid_length = self.config['DATA']['MAX_FRAMES'] * self.config['DATA']['MAX_REGIONS']
        self.max_txt_length = self.config['DATA']['MAX_TXT_LENGTH']
        self.d_vid = self.config['VID_CONFIG']['D_VID']
        self.block_length = self.config['DATA']['BLOCK_LENGTH']
        self.max_eval_threads = self.config['TEST']['MAX_THREADS']
        self.gpu_eval = VidCapModel._check_generation_graph_capability(self.config['TX_CONFIG']['GENERATION'])
        self.cpu_eval = not self.gpu_eval
        self.deterministic = self.seed is not None
        
        if self.deterministic:
            self._set_seed(self.seed)

        tf.get_logger().setLevel('ERROR')
        tf.debugging.set_log_device_placement(False)
        nvidia_smi.nvmlInit()
        self.gpu_handles = [nvidia_smi.nvmlDeviceGetHandleByIndex(n) for n in range(nvidia_smi.nvmlDeviceGetCount())]

        self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=self.cross_device_ops(self.num_packs))
        self.num_repl = self.strategy.num_replicas_in_sync
        print('Successfully allocated {} workers.'.format(self.num_repl))

        with self.strategy.scope():
            self.df = Generator(data_path=self.feat_path,
                                max_vid_length=self.max_vid_length,
                                max_txt_length=self.max_txt_length)

            self.dataset = tf.data.Dataset.from_tensor_slices(self.df.filenames)
            self.dataset = self.dataset.interleave(lambda filename: tf.data.Dataset.from_generator(
                                                self.df, 
                                                (tf.float32, 
                                                 tf.int32,
                                                 tf.int32,
                                                 tf.int32),
                                                (tf.TensorShape([self.max_vid_length,self.d_vid]), 
                                                 tf.TensorShape([self.max_vid_length,]),
                                                 tf.TensorShape([self.max_txt_length,]),
                                                 tf.TensorShape([self.max_txt_length,])),
                                                args=(filename,)),
                                                cycle_length=tf.data.experimental.AUTOTUNE, 
                                                block_length=self.block_length,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                                deterministic=self.deterministic)

            self.dataset = (self.dataset.cache()
                                       .map(self.df.tf_random_mask, 
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                            deterministic=self.deterministic)
                                       .batch(self.batch_size, drop_remainder=True)
                                       .prefetch(tf.data.experimental.AUTOTUNE).repeat(-1)
                            )

            self.dist_dataset = self.strategy.experimental_distribute_dataset(self.dataset)
            self.iter_dataset = iter(self.dist_dataset)
        self.max_warmup_steps = int(self.warmup_ratio * self.max_steps)
        
        if self.lr_anneal:
            self.lr = LRSchedule(max_steps=self.max_steps,
                                 max_steps_warmup=self.max_warmup_steps,
                                 lr_0=self.lr_0,
                                 lr_0_warmup=self.lr_0_warmup)
        else:
            self.lr = self.lr_0

        self.optimizer = Adam(learning_rate=self.lr)

        if self.max_eval_threads is not None and self.max_eval_threads > 0 and self.cpu_eval:
            with tf.device('cpu:0'):
                self.evaluator = VID_CAP_Evaluator(config=config)
            self.queueLock = Lock()
            self.evalQueue = Queue()
            Trainer.Eval_Exit_Flag = 0
        
        elif self.gpu_eval:
            self.evaluator = Evaluator(config=config, strategy=self.strategy)

        self.metrics = {}
        self._init_metrics()
        self.writer = tf.summary.create_file_writer(logdir=os.path.join(self.outputs_path,'train_logs',self.run_name))
        tf.summary.trace_on(graph=False, profiler=False)
        

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
            if mtype=='Captions':
                continue
            for mname in self.metrics[mtype]:
                self.metrics[mtype][mname].reset_states()

    def update_tf_summary(self, step, exclude_caption=True):
        with self.writer.as_default():
            for mtype in self.metrics:
                for mname in self.metrics[mtype]:
                    summary_name = '{}/{}'.format(mtype, mname)
                    summary_result = self.metrics[mtype][mname].result().numpy()
                    if mtype == 'Losses' or mtype == 'Accuracies' or mtype == 'Scalars':
                            tf.summary.scalar(summary_name, summary_result, step=step)
                    elif mtype == 'Distributions':
                            tf.summary.histogram(summary_name, summary_result, step=step)
                            self.metrics[mtype][mname].reset_states()
                    elif mtype == 'Captions' and (not exclude_caption):
                        tf.summary.text(summary_name, summary_result, step=step)

    def pretty_progress(self,
                        step,
                        max_steps,
                        t_step,
                        **metrics):

        eta_secs = min(2**32, int((max_steps - step) * t_step))

        if not hasattr(self, 'ema_eta'):
            self.ema_eta = 0
            self.ema_alpha = 0.01
        
        self.ema_eta = int(self.ema_alpha * eta_secs + (1-self.ema_alpha) * self.ema_eta)

        progress_str = ''
        progress_str += 'Iter {}/{}'.format(step+1,max_steps)
        for metric in metrics:
            if metric == 'lr':
                progress_str += ' - {}: {:0.3e}'.format(metric, metrics[metric])
            elif metric == 'gpu_util':
                progress_str += ' - {}: {:0.1f}'.format(metric, metrics[metric])
            else:
                progress_str += ' - {}: {:0.4f}'.format(metric, metrics[metric])
        
        progress_str += '{:>5} //{:>5} ETA {:0>8}, {:0.2f}/step {:>10}'.format('', '', str(timedelta(seconds=self.ema_eta)), t_step, '')
        progress_str += '\n'
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        logger.info("PROGRESS: {}%".format(round(100*(step+1) / max_steps, 4)))

    def _set_seed(self,
                  seed):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _compute_loss_0(self,
                        labels,
                        predictions,
                        mask=None,
                        reduction=tf.losses.Reduction.AUTO):
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction = reduction)
        loss_ = loss_object(labels, predictions, mask)
        return loss_

    def _init_metrics(self):
        self.add_metric(SparseCategoricalAccuracy(name='Accuracies/Train_Accuracy_CMLM'))
        self.add_metric(MovingAverage(name='Losses/Train_Loss_CMLM'))
        self.add_metric(Scalar(name='Scalars/Train_Learning_Rate'))

    def _compute_loss(self, outputs, labels, task):
        txt_labels = labels['txt_labels']
        txt_labels_mask = labels['txt_labels_attn_mask']
        txt_predictions = outputs['predictions']

        per_example_loss_pred = self._compute_loss_0(labels = txt_labels,
                                                     predictions = txt_predictions,
                                                     mask = txt_labels_mask,
                                                     reduction = tf.keras.losses.Reduction.NONE)

        per_example_loss_pred = tf.reduce_mean(per_example_loss_pred, axis=1) # (batch_size, )
        per_example_loss_model = outputs['loss'] # (batch_size, )
        per_example_loss = per_example_loss_pred + per_example_loss_model * self.lr_r3
        
        avg_loss = tf.reduce_mean(per_example_loss) #(1/n)*L ; n is size of miniBatch
        
        self.metrics['Losses']['Train_Loss_'+task].update_state(avg_loss)
        self.metrics['Accuracies']['Train_Accuracy_'+task].update_state(txt_labels, txt_predictions, txt_labels_mask)

        return avg_loss

    def load_from_h5(self, layer, model_path, ignore_missed=True):
        h5_handle = h5py.File(model_path, 'r')
        missed = []
        for w in layer.weights:
            consumed = assing_weight_from_h5(w, h5_handle)
            if not consumed:
                missed.append(w)
        if (not ignore_missed) and len(missed)>0:
            raise ValueError('Skipped {} weights:\n {}'.format(missed))

    def train(self):
        #note that average over loss is done after applying gradients (k times)
        #hence we should scale lr by k: number of workers/replica
        @tf.function(input_signature=[self.iter_dataset.element_spec])
        def _train_step(dist_inputs):
            def _step_fn(inputs):
                inputs_cmlm, labels_cmlm = prepare_inputs_for_training(inputs, task='causal_masked_language_modeling')
                outputs_cmlm = self.model(**inputs_cmlm)
                    
                loss_cmlm = self._compute_loss(outputs_cmlm, labels_cmlm, 'CMLM')
                loss = loss_cmlm
                    
                weights = self.model.trainable_variables
                gradients = tf.gradients(loss, weights, gate_gradients=self.deterministic)

                #these optimizers apply gradients k times without averaging
                #hence, this k should be compensated in learning rate
                self.optimizer.apply_gradients([[g,w] for g,w in zip(gradients,weights)])
                self.metrics['Scalars']['Train_Learning_Rate'].update_state(self.optimizer._decayed_lr(tf.float32))

                return loss

            per_replica_losses = self.strategy.run(_step_fn, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        with self.strategy.scope():
            ckpt_dir = os.path.join(self.outputs_path,'checkpoints',self.run_name)
            manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                 directory=ckpt_dir,
                                                 max_to_keep=None)
            
            if self.max_eval_threads is not None and self.max_eval_threads > 0 and self.cpu_eval:
                # Create eval threads
                eval_threads = [] 
                for threadID in range(self.max_eval_threads):
                    thread = SampleEvalThread(threadID, self.evalQueue, self.queueLock, self.evaluator.test_cpu) 
                    thread.start() 
                    eval_threads.append(thread)
            elif self.gpu_eval:
                self.evaluator.model = self.model

            print('Training...')
            self.reset_metrics()
            for step in range(self.max_steps):
                #gpu_util = np.mean([nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu for handle in self.gpu_handles])
                ####################
                t_step = time.time()
                dist_inputs = self.iter_dataset.get_next()
                t_step_data = time.time() - t_step
                ####################
                t_step = time.time()
                loss_ = _train_step(dist_inputs)
                t_step_train = time.time() - t_step
                ####################
                if (step+1) % float(self.tb_update_period) == 0:
                    self.update_tf_summary(step=step)
                ####################
                t_step = t_step_data + t_step_train
                self.pretty_progress(step=step,
                                     max_steps=self.max_steps,
                                     t_step=t_step,
                                     loss=self.metrics['Losses']['Train_Loss_CMLM'].result().numpy(),
                                     acc=self.metrics['Accuracies']['Train_Accuracy_CMLM'].result().numpy(),
                                     lr=self.metrics['Scalars']['Train_Learning_Rate'].result().numpy(),
                                     #gpu_util = gpu_util,
                                     t_step_data = t_step_data,
                                     t_step_train = t_step_train,
                                     )
                
                #save a checkpoint every "ckpt_save_period" steps - "inf" means no eval
                if (step+1) % float(self.ckpt_save_period) == 0:
                    print('Saving checkpoint...')
                    ckpt_thd = CkptThread(target=manager.save, args=(step+1,))
                    ckpt_thd.start()
                    
                    if self.max_eval_threads is not None and self.max_eval_threads > 0 and self.cpu_eval:
                        print('Calculating COCO scores in background...')
                        self.queueLock.acquire()
                        self.evalQueue.put({'eval_step': step+1, 
                                            'refresh_ckpt_path': ckpt_thd, 
                                            'silent': True})
                        self.queueLock.release()
                    
                    elif self.gpu_eval:
                        self.evaluator.test_gpu(step+1)
        
        print('Training done.')

        if self.max_eval_threads is not None and self.max_eval_threads > 0 and self.cpu_eval:
            print('Waiting for remaining evaluation threads to finish...')
            # Wait for the queue to empty 
            while not self.evalQueue.empty():
                progress_str = '{} queued eval threads remaining...\r'.format(self.evalQueue.qsize())
                sys.stdout.write(progress_str)
                sys.stdout.flush()
                time.sleep(5)
                pass
              
            # Notify threads it's time to exit 
            Trainer.Eval_Exit_Flag = 1
              
            # Wait for all threads to complete 
            for t in eval_threads: 
                t.join() 
            
            print ("All evaluation threads finished.") 
                    

class VID_CAP_Trainer(Trainer):
    def __init__(self,
                 config,
                 **kwargs):

        super(VID_CAP_Trainer, self).__init__(config=config, **kwargs)
        with self.strategy.scope():
            print('Building video model...')
            self.model = VidCapModel(config=self.config)
            self.checkpoint = tf.train.Checkpoint(model=self.model)

            print('Initializing variables...')
            self.model.set_strategy(self.strategy)
            self.model.distributed_init(batch_size=self.batch_size, d_vid=self.d_vid)

            if self.pre_train:
                print('Loading variables from checkpoint...')
                self.checkpoint.restore(self.pretrain_ckpt_path)#.assert_consumed()

            if (
                self.config['TX_CONFIG']['ENCODER']['INITIALIZATION']
                or self.config['TX_CONFIG']['DECODER']['INITIALIZATION']
                ):

                print('Loading transformer variables from h5 pre-trained weights...')
                skipped = self.model.tx.load_pre_trained()
                if len(skipped)>0:
                    print('{} weights were skipped:'.format(len(skipped)))
                    for w in skipped:
                        print(w.name)

            if self.gpu_eval:
                self.evaluator.model = self.model
        print('All initializations done.')
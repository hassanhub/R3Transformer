import tensorflow as tf
import re, h5py
import numpy as np
from tensorflow.keras.layers import Layer, Dropout, Dense
from tensorflow.keras import Model
from transformers import T5Tokenizer
from tx_helper import EncoderDecoder

class VidCap_Base(Model):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def set_strategy(self,
                     strategy):
        self.strategy = strategy

    @tf.function
    def _init_0(self,
                vid_inputs, 
                txt_inputs, 
                vid_inputs_attn_mask, 
                txt_inputs_attn_mask):
        
        self.call(vid_inputs=vid_inputs, 
                  txt_inputs=txt_inputs, 
                  vid_inputs_attn_mask=vid_inputs_attn_mask, 
                  txt_inputs_attn_mask=txt_inputs_attn_mask,
                  training=False)
    
    @tf.function
    def _init_d(self,
                vid_inputs, 
                txt_inputs, 
                vid_inputs_attn_mask, 
                txt_inputs_attn_mask):
        self.strategy.run(self._init_0,args=(vid_inputs, 
                                             txt_inputs, 
                                             vid_inputs_attn_mask, 
                                             txt_inputs_attn_mask,))

    def _dummy_inputs(self,
                      batch_size,
                      vid_inputs=None,
                      from_features=True,
                      d_vid=0):
        txt_max_length = 1
        vid_max_length = 1
        if from_features:
            vid_inputs = tf.random.uniform([batch_size,1,d_vid], minval=0, maxval=255, dtype=tf.float32)
        else:
            vid_inputs = tf.cast(vid_inputs, dtype=tf.uint8)

        txt_inputs = tf.random.uniform([batch_size,1], minval=0, maxval=10, dtype=tf.int32)
        vid_inputs_attn_mask = tf.random.uniform([batch_size,1], minval=0, maxval=1, dtype=tf.int32)
        txt_inputs_attn_mask = tf.random.uniform([batch_size,1], minval=0, maxval=1, dtype=tf.int32)
        return vid_inputs, txt_inputs, vid_inputs_attn_mask, txt_inputs_attn_mask

    def init(self,
             batch_size,
             vid_inputs=None,
             from_features=True,
             d_vid=0):
        (vid_inputs, 
         txt_inputs, 
         vid_inputs_attn_mask, 
         txt_inputs_attn_mask,
         ) = self._dummy_inputs(batch_size, vid_inputs, from_features, d_vid)
        self._init_0(vid_inputs, txt_inputs, vid_inputs_attn_mask, txt_inputs_attn_mask)

    def distributed_init(self,
                         batch_size,
                         vid_inputs=None,
                         from_features=True,
                         d_vid=0):
        (vid_inputs, 
         txt_inputs, 
         vid_inputs_attn_mask, 
         txt_inputs_attn_mask,
         ) = self._dummy_inputs(batch_size, vid_inputs, from_features, d_vid)
        self._init_d(vid_inputs, txt_inputs, vid_inputs_attn_mask, txt_inputs_attn_mask)

    def freeze_decoder(self,
                       skip_embeddings=True):
        self.tx.decoder.trainable = False
        self.tx.decoder_embeddings.trainable = skip_embeddings
        
class VidCapModel(VidCap_Base):
    def __init__(self,
                 config,
                 name='VidCapModel',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        config_tx = config['TX_CONFIG']
        config_tx['GENERATION']['MAX_LENGTH'] = config['DATA']['MAX_TXT_LENGTH']
        config_tx['ENCODER']['MAX_SPT_POSITIONS'] = config['DATA']['MAX_REGIONS']
        config_tx['ENCODER']['MAX_TMP_POSITIONS'] = config['DATA']['MAX_FRAMES']
        config_tx['ENCODER']['ROLE_TYPES'] = eval(config_tx['ENCODER']['ROLE_TYPES'])
        config_tx['DECODER']['MAX_TMP_POSITIONS'] = config['DATA']['MAX_TXT_LENGTH']
        config_tx['DECODER']['SELF_ROLE_TYPES'] = eval(config_tx['DECODER']['SELF_ROLE_TYPES'])
        config_tx['DECODER']['CROSS_ROLE_TYPES'] = eval(config_tx['DECODER']['CROSS_ROLE_TYPES'])

        self.gen_cfg = config_tx['GENERATION']
        self.gen_fn = None

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.tx = EncoderDecoder(config_tx,
                                 name='transformer')

        self.proj = Dense(self.tx.encoder.d_model,
                          use_bias=False,
                          name='vid_proj')


    @staticmethod
    def _check_generation_graph_capability(gen_cfg):
        use_cache = gen_cfg['USE_CACHE']
        num_beams = gen_cfg['NUM_BEAMS']
        length_penalty = gen_cfg['LENGTH_PENALTY']
        repetition_penalty = gen_cfg['REPETITION_PENALTY']
        no_repeat_ngram_size = gen_cfg['NO_REPEAT_NGRAM_SIZE']
        
        autograph = True
        autograph = autograph and (not use_cache)
        autograph = autograph and (num_beams == 1)
        autograph = autograph and (length_penalty == 1.0)
        autograph = autograph and (repetition_penalty == 1.0)
        autograph = autograph and (no_repeat_ngram_size == 0)
        
        return autograph

    @tf.function
    def get_vid_features(self, 
                         vid_inputs, 
                         training):
        
        vid_pos_ids = vid_inputs[:,:,-5:]
        vid_features = vid_inputs[:,:,:-5]
        vid_features = self.proj(vid_features)
        vid_features = tf.nn.relu(vid_features)

        return vid_features, vid_pos_ids

    @tf.function
    def call(self,
             vid_inputs=None,
             txt_inputs=None,
             vid_inputs_attn_mask=None,
             txt_inputs_attn_mask=None,
             training=False):


        vid_features, vid_pos_ids = self.get_vid_features(vid_inputs, training)

        outputs = self.tx(inputs=None,
                          inputs_embeds=vid_features,
                          attention_mask=vid_inputs_attn_mask,
                          encoder_pos_ids=vid_pos_ids,
                          decoder_input_ids=txt_inputs,
                          decoder_attention_mask=txt_inputs_attn_mask,
                          use_cache=False,
                          training=training)
        
        logits = outputs['decoder_head_logits']
        classes = tf.nn.softmax(logits)
        loss = outputs['loss']
        outputs.update({'predictions': classes, 'loss': loss})
        return outputs

    @tf.function
    def generate(self,
                 vid_inputs, 
                 vid_mask,
                 **kwargs):

        vid_features, vid_pos_ids = self.get_vid_features(vid_inputs, False)
        
        if self.gen_fn is None:
            #initialize self.gen_fn
            self.autograph = VidCapModel._check_generation_graph_capability(self.gen_cfg)
            if self.autograph:
                self.gen_fn = tf.function(func=self.tx.generate, experimental_autograph_options=tf.autograph.experimental.Feature.EQUALITY_OPERATORS)
            else:
                self.gen_fn = self.tx.generate

        decoded = self.gen_fn(input_embeds=vid_features,
                              attention_mask=vid_mask,
                              encoder_pos_ids=vid_pos_ids,
                              max_length=self.gen_cfg['MAX_LENGTH'],
                              early_stopping=self.gen_cfg['EARLY_STOPPING'],
                              num_beams=self.gen_cfg['NUM_BEAMS'],
                              no_repeat_ngram_size=self.gen_cfg['NO_REPEAT_NGRAM_SIZE'],
                              repetition_penalty=self.gen_cfg['REPETITION_PENALTY'],
                              length_penalty=self.gen_cfg['LENGTH_PENALTY'],
                              pad_token_id=self.gen_cfg['PAD_TOKEN_ID'],
                              eos_token_id=self.gen_cfg['EOS_TOKEN_ID'],
                              decoder_start_token_id=self.gen_cfg['DECODER_START_TOKEN_ID'],
                              use_cache=self.gen_cfg['USE_CACHE'],
                              autograph=self.autograph,
                             **kwargs)
        
        return decoded
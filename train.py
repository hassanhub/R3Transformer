import argparse, yaml, os
from trainers import VID_CAP_Trainer

def replace_value(obj, key, value):
    keys = [k.upper() for k in key.split('/')]
    for key in keys[:-1]:
        assert (obj is not None) and (key in obj), "Arg not found in config file!"
        obj = obj.get(key)
    assert (obj is not None) and (keys[-1] in obj), "Arg not found in config file!"
    assert not isinstance(obj[keys[-1]], dict), "Please specify the desired arg within current sub_config."
    orig_type = type(obj[keys[-1]])
    if orig_type == int:
        value = eval(value)
    obj[keys[-1]] = orig_type(value)

def main(config):
    trainer = VID_CAP_Trainer(config = config)
    trainer.train()

if __name__ == '__main__':
    description = 'The main train call function.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config_path', type=str, default='./configs/VQ_2D3D_T5_base.yml')
    parser.add_argument('--outputs_path', type=str, default='./outputs')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    config['TRAIN']['OUTPUTS_PATH'] = args.outputs_path
    if args.data_path != './data':
        os.system('ln -s {} ./data'.format(args.data_path))
    optional_args = args.opts
    for arg in optional_args:
        key, value = arg.split(':')
        replace_value(config, key, value)

    main(config)
import argparse
from eval.glue_eval import inference as glue_inference
from eval.task_transfer_eval import inference as task_inference
from train.hf_train_peft import get_config
from collections import namedtuple
from train.utils import get_dataset_option, is_multilingual


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config_path', type=str,
                        default='./configs/eval_config.yaml')
    args = parser.parse_args()

    config = get_config(args.config_path)
    config = namedtuple('config', config.keys())(*config.values())

    DatasetOption = get_dataset_option(config.model_name)

    if 'glue' in args.dataset:
        dataloader = DatasetOption.get('mixture')(
            args.dataset, split='validation')
        glue_inference(
            model_name=config.model_name,
            config=config,
            dataloader=dataloader,
            eval_name=args.dataset
        )
    else:
        if is_multilingual(config.model_name):
            dataloader = DatasetOption.get(args.dataset)(
                language=config.language, split='validation')
        else:
            dataloader = DatasetOption.get(args.dataset)(
                split='validation')
        task_inference(
            model_name=config.model_name,
            config=config,
            dataloader=dataloader,
            eval_name=args.dataset
        )

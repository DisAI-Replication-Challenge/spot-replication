import argparse
from eval.glue_eval import inference as glue_inference
from data.t5.task import DatasetOption
from train.hf_train_peft import get_config
from collections import namedtuple


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config_path', type=str,
                        default='./configs/eval_config.yaml')
    args = parser.parse_args()

    dataloader = DatasetOption.get('mixture')(args.dataset)
    config = get_config(args.config_path)
    config = namedtuple('config', config.keys())(*config.values())

    glue_inference(
        model_name=config.model_name,
        config=config,
        dataloader=dataloader,
        eval_name=args.dataset
    )

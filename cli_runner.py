import yaml
import argparse
from hf_lib.trainer import Trainer
from hf_lib.predictor import Predictor


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    predict_parser = subparsers.add_parser("predict")

    train_parser.add_argument("--config", type=str)
    train_parser.add_argument("--data_path", type=str)
    train_parser.add_argument("--out_path", type=str)
    predict_parser.add_argument("--config", type=str)
    predict_parser.add_argument("--data_path", type=str)
    predict_parser.add_argument("--out_path", type=str)

    args = parser.parse_args()
    with open(args.config) as c:
        config = yaml.safe_load(c)

    if args.command == "train":
        model_arch = config['HF']['model_arch']
        model_name = config['HF']['model_name']
        if args.data_path:
            data_path = args.data_path
        else:
            data_path = config['CONFIG']['data_path']
        if args.out_path:
            out_path = args.out_path
        else:
            out_path = config['CONFIG']['out_path']
        epochs = int(config['CONFIG']['epochs'])
        batch_size = int(config['CONFIG']['batch_size'])
        test_size = float(config['CONFIG']['test_size'])

        mytrainer = Trainer(model_arch=model_arch,
                            model_name=model_name,
                            data_path=data_path,
                            out_path=out_path,
                            epochs=epochs,
                            batch_size=batch_size,
                            test_size=test_size
                            )
        mytrainer.run()
    elif args.command == "predict":
        model_arch = config['HF']['model_arch']
        model_name = config['HF']['model_name']
        state_dict_path = config['CONFIG']['state_dict_path']
        labels = config['CONFIG']['labels']
        text_col = int(config['CONFIG']['text_col'])
        if args.data_path:
            data_path = args.data_path
        else:
            data_path = config['CONFIG']['data_path']
        if args.out_path:
            out_path = args.out_path
        else:
            out_path = config['CONFIG']['out_path']
        batch_size = int(config['CONFIG']['batch_size'])

        mypredictor = Predictor(model_arch=model_arch,
                                model_name=model_name,
                                state_dict_path=state_dict_path,
                                labels=labels,
                                text_col=text_col,
                                data_path=data_path,
                                out_path=out_path,
                                batch_size=batch_size
                                )
        mypredictor.run()


if __name__ == '__main__':
    main()

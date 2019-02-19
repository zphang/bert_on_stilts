from sklearn.metrics import matthews_corrcoef, f1_score

import argparse
import json
import os
import pandas as pd

from scipy.stats import pearsonr, spearmanr

import glue.tasks as tasks


PROCESSORS = {
    "cola": tasks.ColaProcessor,
    "sst": tasks.SstProcessor,
    "mrpc": tasks.MrpcProcessor,
    "stsb": tasks.StsbProcessor,
    "qqp": tasks.QqpProcessor,
    "mnli": tasks.MnliProcessor,
    "mnli-mm": tasks.MnliMismatchedProcessor,
    "qnli": tasks.QnliProcessor,
    "rte": tasks.RteProcessor,
    "wnli": tasks.WnliProcessor,
    "xnli": tasks.XnliProcessor,
    "snli": tasks.SnliProcessor,
    "bcs": tasks.BcsProcessor,
}
OUTPUT_MODES = {
    "cola": "classification",
    "sst": "classification",
    "mrpc": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "xnli": "classification",
    "snli": "classification",
    "bcs": "classification",
}
DEFAULT_FOL_NAMES = {
    "cola": "CoLA",
    "sst": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
}


def simple_accuracy(pred_srs, label_srs):
    return (pred_srs == label_srs).mean()


def acc_and_f1(pred_srs, label_srs):
    acc = simple_accuracy(pred_srs, label_srs)
    f1 = f1_score(y_true=label_srs, y_pred=pred_srs)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(pred_srs, label_srs):
    pearson_corr = float(pearsonr(pred_srs, label_srs)[0])
    spearman_corr = float(spearmanr(pred_srs, label_srs)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, pred_srs, label_srs):
    assert len(pred_srs) == len(label_srs)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(label_srs, pred_srs)}
    elif task_name == "sst":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "mrpc":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "stsb":
        return pearson_and_spearman(pred_srs, label_srs)
    elif task_name == "qqp":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    else:
        raise KeyError(task_name)


def load_labels(task_name, data_dir):
    processor = PROCESSORS[task_name]()
    examples = processor.get_dev_examples(data_dir)
    output_mode = OUTPUT_MODES[task_name]
    if output_mode == "classification":
        label2idx = {label: num for (num, label) in enumerate(processor.get_labels())}
        label_srs = pd.Series([label2idx[example.label] for example in examples])
    elif output_mode == "regression":
        label_srs = pd.Series([example.label for example in examples])
    else:
        raise KeyError(output_mode)
    return label_srs


def load_preds(task_name, pred_file_path):
    pred_df = pd.read_csv(pred_file_path, header=None, sep="\t")
    output_mode = OUTPUT_MODES[task_name]
    if output_mode == "classification":
        pred_srs = pred_df.idxmax(axis=1)
    elif output_mode == "regression":
        pred_srs = pred_df[0]
    else:
        raise KeyError(output_mode)
    return pred_srs


def compute_metrics_from_paths(task_name, pred_file_path, task_data_dir):
    pred_srs = load_preds(task_name, pred_file_path)
    label_srs = load_labels(task_name, task_data_dir)
    return compute_metrics(task_name, pred_srs, label_srs)


def get_default_task_data_dir(task_name):
    glue_path = os.environ["GLUE_DIR"]
    return os.path.join(glue_path, DEFAULT_FOL_NAMES[task_name])


def main():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--task-name', required=True)
    parser.add_argument('--pred-file-path', required=True)
    parser.add_argument('--task-data-dir', default=None)
    parser.add_argument('--no-print', action="store_true")
    parser.add_argument('--output-path', required=False, default=None)
    args = parser.parse_args()
    task_name = args.task_name.lower()
    if args.task_data_dir is None:
        task_data_dir = get_default_task_data_dir(args.task_name)
    else:
        task_data_dir = args.task_data_dir
    metrics = compute_metrics_from_paths(task_name, args.pred_file_path, task_data_dir)
    if not args.no_print:
        print(metrics)
    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            f.write(json.dumps(metrics, indent=2) + "\n")

    # Hack for MNLI-mm
    if task_name == "mnli":
        mm_task_name = "mnli-mm"
        metrics = compute_metrics_from_paths(
            mm_task_name, args.pred_file_path.replace("val_results", "mm_val_results"), task_data_dir,
        )
        if not args.no_print:
            print(metrics)
        if args.output_path is not None:
            with open(args.output_path.replace("val_metrics", "mm_val_metrics"), "w") as f:
                f.write(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()

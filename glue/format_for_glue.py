import argparse
import os
import numpy as np
import pandas as pd

from glue.evaluate import OUTPUT_MODES, PROCESSORS

OUTPUT_NAMES = {
    "cola": "CoLA",
    "sst": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "qnli": "QNLI",
    "rte": "RTE",
}


def read_file(tsv_path, file_format):
    sep_dict = {
        "tsv": "\t",
        "csv": ",",
    }
    return pd.read_csv(tsv_path, sep=sep_dict[file_format], header=None)


def write_srs(srs, output_path):
    output_df = pd.DataFrame({"index": np.arange(len(srs)), "prediction": srs.values})
    return output_df.to_csv(output_path, sep="\t", index=False)


def format_preds(task_name, input_path, output_path, file_format):
    output_mode = OUTPUT_MODES[task_name]
    processor = PROCESSORS[task_name]()
    df = read_file(input_path, file_format)
    if output_mode == "classification":
        label_dict = dict(enumerate(processor.get_labels()))
        output_srs = df.idxmax(axis=1).replace(label_dict)
    elif output_mode == "regression":
        output_srs = df[0]
    else:
        raise KeyError(output_mode)
    write_srs(
        srs=output_srs,
        output_path=output_path,
    )


def format_task(task_name, input_base_path, output_base_path, file_format):
    if task_name == "mnli-mm":
        input_path = os.path.join(input_base_path, "mnli", "mm_test_preds.{}".format(file_format))
    else:
        input_path = os.path.join(input_base_path, task_name, "test_preds.{}".format(file_format))
    format_preds(
        task_name=task_name,
        input_path=input_path,
        output_path=os.path.join(output_base_path, "{}.tsv".format(OUTPUT_NAMES[task_name])),
        file_format=file_format,
    )


def format_all_tasks(input_base_path, output_base_path, file_format):
    for task_name in OUTPUT_NAMES:
        try:
            format_task(
                task_name=task_name,
                input_base_path=input_base_path,
                output_base_path=output_base_path,
                file_format=file_format,
            )
        except FileNotFoundError:
            print("Skipping {}".format(task_name))


def main(task_name, input_base_path, output_base_path, file_format):
    if task_name is None:
        format_all_tasks(
            input_base_path=input_base_path,
            output_base_path=output_base_path,
            file_format=file_format,
        )
    else:
        format_preds(
            task_name=task_name,
            input_path=os.path.join(input_base_path, "test_preds.{}".format(file_format)),
            output_path=os.path.join(output_base_path, "{}.tsv".format(OUTPUT_NAMES[task_name])),
            file_format=file_format,
        )
        if task_name == "mnli":
            format_preds(
                task_name="mnli-mm",
                input_path=os.path.join(input_base_path, "mm_test_preds.{}".format(file_format)),
                output_path=os.path.join(output_base_path, "{}.tsv".format(OUTPUT_NAMES["mnli-mm"])),
                file_format=file_format,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glue')
    parser.add_argument('-t', '--task-name', type=str, default=None)
    parser.add_argument('-i', '--input-base-path', required=True)
    parser.add_argument('-o', '--output-base-path', required=True)
    parser.add_argument('-f', '--file-format', type=str, default="csv")
    args = parser.parse_args()
    main(
        task_name=args.task_name,
        input_base_path=args.input_base_path,
        output_base_path=args.output_base_path,
        file_format=args.file_format,
    )

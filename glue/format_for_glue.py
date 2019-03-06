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


def read_file(tsv_path, file_format="tsv"):
    sep_dict = {
        "tsv": "\t",
        "csv": ",",
    }
    return pd.read_csv(tsv_path, sep=sep_dict[file_format], header=None)


def write_srs(srs, output_path):
    output_df = pd.DataFrame({"index": np.arange(len(srs)), "prediction": srs.values})
    return output_df.to_csv(output_path, sep="\t", index=False)


def format_task(task_name, input_base_path, output_base_path, file_format="tsv"):
    output_mode = OUTPUT_MODES[task_name]
    processor = PROCESSORS[task_name]()
    if task_name == "mnli-mm":
        input_path = os.path.join(input_base_path, "mnli", "mm_test_results.{}".format(file_format))
    else:
        input_path = os.path.join(input_base_path, task_name, "test_results.{}".format(file_format))
    print(input_path)
    df = read_file(input_path, file_format)
    if output_mode == "classification":
        label_dict = dict(enumerate(processor.get_labels()))
        output_srs = df.idxmax(axis=1).replace(label_dict)
    elif output_mode == "regression":
        output_srs = df[0]
    else:
        raise KeyError(output_mode)
    write_srs(
        output_srs,
        os.path.join(output_base_path, "{}.tsv".format(OUTPUT_NAMES[task_name]))
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


def main(task_name, input_path, output_path, file_format):
    if task_name is None:
        format_all_tasks(
            input_base_path=input_path,
            output_base_path=output_path,
            file_format=file_format,
        )
    else:
        format_task(
            task_name=task_name,
            input_path=input_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glue')
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--file-format', type=str, default="tsv")
    args = parser.parse_args()
    main(
        task_name=task_name,
        input_path=args.input_path,
        output_path=args.output_path,
        file_format=args.file_format,
    )

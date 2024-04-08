import os
import sys
import argparse

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor
from loguru import logger
from tqdm import tqdm
import jieba
import snownlp
import jiagu

from utils import Config, Report

MODEL_TOOL = [
    "nlp_structbert_word-segmentation_chinese-base",
    "nlp_structbert_word-segmentation_chinese-base-ecommerce",
    "nlp_structbert_word-segmentation_chinese-lite",
    "nlp_structbert_word-segmentation_chinese-lite-ecommerce",
    "nlp_lstmcrf_word-segmentation_chinese-ecommerce",
    "nlp_lstmcrf_word-segmentation_chinese-news",
]

ALGO_TOOL = [
    "jieba",
    "snownlp",
    "jiagu",
]


def multi_time():
    pass


def one_time(config, input_data=None):
    """
    Run the tools one time.

    Args:
        config (dict): The config.
        input_data (str): The input text.

    Returns:
        results (dict): The results of the tools.
    """
    if input_data is None:
        # waiting for input
        logger.info("Please input the text you want to segment")
        input_data = input()
    logger.info(f"input_data: {input_data}")
    results = dict()
    for target in config["tools"]:
        if target in ALGO_TOOL:
            result = run_algo(config, target, input_data)
            results[target] = result
        elif target in MODEL_TOOL:
            model_name_or_path = os.path.join(config["path"], target)
            result = run_model(input_data, model_name_or_path)
            results[target] = result
        else:
            # Add the error to result
            results[target] = f"target: {target} is not supported"
    # logger.info(f"results: {results}")
    # loop the results and print the result line by line
    for target, result in results.items():
        logger.info(f"target: {target}")
        logger.info(f"result: {result}")
    return results


def evaluation(config):
    """
    Run the tools for evaluation.

    Args:
        config (dict): The config.

    Returns:
        results (dict): The results of the tools.
    """
    logger.info("Start the evaluation")
    report = Report()
    dataset_path = config["dataset_path"]
    dataset_name = config["dataset_name"]
    for dataset in dataset_name:
        with open(os.path.join(dataset_path, "test_" + dataset + ".txt"), "r") as f:
            can_all = f.readlines()
        with open(os.path.join(dataset_path, dataset + ".txt"), "r") as f:
            ref_all = f.readlines()
        logger.info(f"Current Dataset: {dataset}")
        for target in config["tools"]:
            logger.info(f"Target Tool: {target}")
            if target in MODEL_TOOL:
                model_name_or_path = os.path.join(config["path"], target)
                model = Model.from_pretrained(model_name_or_path)
                tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
                pipeline_ins = pipeline(
                    task=Tasks.word_segmentation, model=model, preprocessor=tokenizer
                )
            for ref, can in tqdm(zip(ref_all, can_all), desc="Processing"):
                ref_words = ref.strip().split()
                can_words = get_result(config, target, can, pipeline_ins)
                ref_words_len, can_words_len, acc_word_len = report.compare_line(
                    ref_words, can_words
                )
                report.add_result(
                    dataset=dataset,
                    target=target,
                    ref_words_len=ref_words_len,
                    can_words_len=can_words_len,
                    acc_word_len=acc_word_len,
                )
            report.output_results()
    report.output_results()
    return report.get_results()


def get_result(config, target, line, pipeline_ins=None):
    """
    Get the result of the target tool.

    Args:
        config (dict): The config.
        target (str): The target tool.
        line (str): The input text.
        pipeline_ins (Pipeline): The pipeline.
    Returns:
        result (list): The result of the tool.
    """
    if target in ALGO_TOOL:
        result = run_algo(config, target, line)
    elif target in MODEL_TOOL:
        result = run_pipeline(line, pipeline_ins)
        # result = run_model(line, model_name_or_path)
    else:
        # Add the error to result
        result = f"target: {target} is not supported"

    return result


def run_pipeline(test_data, pipeline):
    """
    Run the pipeline with the test data.

    Args:
        test_data (str): The input text.
        pipeline (Pipeline): The pipeline.

    Returns:
        result (list): The result of the pipeline.
    """
    if isinstance(test_data, str):
        test_data = [test_data]
    result = pipeline(input=test_data)[0]["output"]
    return result


def run_model(
    test_data,
    model_name_or_path="damo/nlp_structbert_word-segmentation_chinese-base",
):
    """
    Run the model with the test data.

    Args:
        test_data (str): The input text.
        model_name_or_path (str): The model name or path.

    Returns:
        result (list): The result of the model.
    """
    if isinstance(test_data, str):
        test_data = [test_data]
    model = Model.from_pretrained(model_name_or_path)
    tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
    pipeline_ins = pipeline(
        task=Tasks.word_segmentation, model=model, preprocessor=tokenizer
    )
    result = pipeline_ins(input=test_data)[0]["output"]
    # logger.debug(f"type: {type(result)}")
    return result


def run_algo(config, algo="jieba", test_data=None):
    if test_data is None:
        test_data = "今天天气不错，适合出去游玩"
    # logger.info(f"test_data: {test_data}")
    # logger.info(f"algo: {algo}")
    result = []
    if algo == "jieba":
        result = jieba.cut(test_data, cut_all=False)
        result = list(result)
    elif algo == "hanlp":
        for term in pyhanlp.HanLP.segment(test_data):
            result.append(term.word)
    elif algo == "snownlp":
        result = snownlp.SnowNLP(test_data).words
    elif algo == "jiagu":
        result = jiagu.seg(test_data)
    else:
        raise ValueError(f"algo: {algo} is not supported")
    return result


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
    )
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="今天天气不错，适合出去游玩")
    args = parser.parse_args()
    config = Config(
        config_file_list=["props/test.yaml", "props/run.yaml"],
    )
    # if config["mode"] == "multi_time":
    #     multi_time(config)
    if config["mode"] == "one_time":
        one_time(config, args.input)
    elif config["mode"] == "eva":
        evaluation(config)
    else:
        raise ValueError(f"mode: {config['mode']} is not supported")

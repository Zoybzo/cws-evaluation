import os

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor
from loguru import logger
import jieba

from utils import Config


def test_algo(config, algo="jieba", test_data=None):
    if test_data is None:
        test_data = "今天天气不错，适合出去游玩"
    logger.info(f"test_data: {test_data}")
    logger.info(f"algo: {algo}")
    if algo == "jieba":
        seg_list = jieba.cut(test_data, cut_all=False)
        logger.info(f"Default Mode: {'/'.join(seg_list)}")
    else:
        raise ValueError(f"algo: {algo} is not supported")


def test_model(
    test_data,
    model_name_or_path="damo/nlp_structbert_word-segmentation_chinese-base",
):
    if test_data is None:
        test_data = "今天天气不错，适合出去游玩"
        # {'output': '今天 天气 不错 ， 适合 出去 游玩'}
    logger.info(f"test_data: {test_data}")
    if isinstance(test_data, str):
        test_data = [test_data]
    logger.info(f"model_name_or_path: {model_name_or_path.split('/')[-1]}")
    model = Model.from_pretrained(model_name_or_path)
    tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
    pipeline_ins = pipeline(
        task=Tasks.word_segmentation, model=model, preprocessor=tokenizer
    )
    result = pipeline_ins(input=test_data)
    print(result)


def test_models(config, test_data=None):
    for model_name_or_path in config["models"]:
        model_name_or_path = os.path.join(config["path"], model_name_or_path)
        test_model(test_data, model_name_or_path)


if __name__ == "__main__":
    config = Config(config_file_list=["props/test.yaml"])
    test_models(config)
    test_algo(config)

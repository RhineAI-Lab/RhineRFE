from modelscope.msdatasets import MsDataset

ds = MsDataset.load('AI-ModelScope/sharegpt_gpt4', cache_dir='../dataset')

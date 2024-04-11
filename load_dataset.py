import datasets
import torchtext


class LoadDataSet():
    """
    Load datasets using 'datasets' library. We will use 'imdb' datasets from :
    https://huggingface.co/datasets/stanfordnlp/imdb
    """

    def __init__(self) -> None:
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    
    def load_datasets(self):
        """
        Load 'imdb' datasets then return train and test data

        Return 
        ------
        train_data:
            train data from 'imdb' datasets. This table includes :
                -text: string format 
                -label: classification label with possibles values includings neg(0) and pos(1)
        test_data:
            test data from 'imdb' datasets. This table includes :
                -text: string format 
                -label: classification label with possibles values includings neg(0) and pos(1)
        """
        train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

        def tokenize_example(example, tokenizer, max_length):
            tokens = tokenizer(example["text"])[:max_length]
            return {"tokens": tokens}

        max_length = 256

        train_data = train_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )
        test_data = test_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )

        return train_data, test_data
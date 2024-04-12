import torchtext
import torch
import torch.nn as nn

from load_dataset import LoadDataSet

class PreProcessing(LoadDataSet):

    def __init__(self) -> None:
        super().__init__()
        self.train_data, self.test_data = self.load_datasets()
        print(self.train_data)
        print(self.test_data)
        print(self.train_data.features)
        self.train_data[0]["tokens"][:25]
        self.valid_data = None
        self.unk_index = None
        self.pad_index = None
        self.vocab = None

    def creat_validation_data(self):
        print("je suis dans la creation de data")
        test_size = 0.25
        train_valid_data = self.train_data.train_test_split(test_size=test_size)
        self.train_data = train_valid_data["train"]
        self.valid_data = train_valid_data["test"]
        print(len(self.train_data))
        print(len(self.valid_data))
        print(len(self.test_data))

    def create_vocabulary(self):
        
        # Create validation data to split the datasets between train and valid data
        self.creat_validation_data()

        min_freq = 5
        special_tokens = ["<unk>", "<pad>"]

        vocab = torchtext.vocab.build_vocab_from_iterator(
            self.train_data["tokens"],
            min_freq=min_freq,
            specials=special_tokens,
        )
        self.unk_index = vocab["<unk>"]
        self.pad_index = vocab["<pad>"]

        print(len(vocab))
        print(vocab.get_itos()[:10])
        print(vocab["and"])

        # Useful to not raise an error but just an integer (here 0 as unk_index is vocab index 0)
        vocab.set_default_index(self.unk_index)

        print(vocab.lookup_indices(["hello", "world", "some_token", "<pad>"]))
        print(vocab.lookup_indices(["hello", "world", "some_token", "<pad>"]))
        return vocab

    def vocab_to_numerical_data(self, vocab):
        def numericalize_example(example, vocab):
            ids = vocab.lookup_indices(example["tokens"])
            return {"ids": ids}

        self.train_data = self.train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
        self.valid_data = self.valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
        self.test_data = self.test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

        # with_format delete columns not specified (pytorch works only with integer)
        self.train_data = self.train_data.with_format(type="torch", columns=["ids", "label"])
        self.valid_data = self.valid_data.with_format(type="torch", columns=["ids", "label"])
        self.test_data = self.test_data.with_format(type="torch", columns=["ids", "label"])

    def create_dataloader(self):

        def get_collate_fn(pad_index):
            def collate_fn(batch):
                batch_ids = [i["ids"] for i in batch]
                batch_ids = nn.utils.rnn.pad_sequence(
                    batch_ids, padding_value=pad_index, batch_first=True
                )
                batch_label = [i["label"] for i in batch]
                batch_label = torch.stack(batch_label)
                batch = {"ids": batch_ids, "label": batch_label}
                return batch

            return collate_fn

        def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
            collate_fn = get_collate_fn(pad_index)
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
            )
            return data_loader

        batch_size = 512
        # Numerize vocab data
        self.vocab = self.create_vocabulary()
        self.vocab_to_numerical_data(vocab=self.vocab)

        train_data_loader = get_data_loader(self.train_data, batch_size, self.pad_index, shuffle=True)
        valid_data_loader = get_data_loader(self.valid_data, batch_size, self.pad_index)
        test_data_loader = get_data_loader(self.test_data, batch_size, self.pad_index)

        return train_data_loader, valid_data_loader, test_data_loader
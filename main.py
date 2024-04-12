import numpy as np
import os
import torch

from build_model import TrainModel

if __name__ == "__main__":
    
    ##########################################
    # Working exemple.
    seed = 1234
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    ##########################################

    trained_model = TrainModel()
    print("ici")

    # Data to load for training
    train_data_loader = trained_model.train_data_loader
    valid_data_loader = trained_model.valid_data_loader
    model = trained_model.model
    optimizer = trained_model.optimizer
    criterion = trained_model.criterion
    device = trained_model.device

    if not os.path.isfile("nbow.pt"):
        trained_model.train_model(
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

    model.load_state_dict(torch.load("nbow.pt"))

    test_data_loader = trained_model.test_data_loader

    tokenizer = trained_model.tokenizer
    vocab = trained_model.vocab

    test_loss, test_acc = trained_model.evaluate(test_data_loader, model, criterion, device)
    print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

    def predict_sentiment(text, model, tokenizer, vocab, device):
        tokens = tokenizer(text)
        ids = vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = model(tensor).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
        return predicted_class, predicted_probability

text = "This film is terrible!"

print(predict_sentiment(text, model, tokenizer, vocab, device))

text = "This film is great!"

print(predict_sentiment(text, model, tokenizer, vocab, device))

text = "This film is not terrible, it's great!"

print(predict_sentiment(text, model, tokenizer, vocab, device))

text = """This is what Hollywood needs. A great story with a great director/producer. After that the best thing a studio can do is get the hell out of the way and let artists create art.

Dune Part 2 is creative, beautiful, tragic, and mesmerizing. Never once did I get bored or anticipate what was going to happen next. I haven't read the book so I have nothing to compare it to.

Denis Villeneuve continues to amaze me with the effort he puts into each of his films. The acting in this film was top notch too. We saw it in IMAX and the sound was earth shattering. If you're gonna see this movie, see it on the largest screen possible."""

print(predict_sentiment(text, model, tokenizer, vocab, device))

text = """I feel like I'm taking crazy pills or have missed a dose, as I cannot comprehend the high praise for this sequel.

I adored Part 1, which introduced me to this trippy Sci-Fi and the grandeur of a far-flung potential future.

Part 2 was all spectacle, yelling, and repetition (we get it, he's the chosen one), marred by:

poor editing (I'm genuinely shocked they didn't start the film on Giedi Prime with that battle sequence. Beginning it there would have pulled us in to this weird SciFi world and made it clear that there was some horrific villain who would be a huge threat to Paul and his ambitions, instead we got a slow paced intro about returning a dead body. Dull)

Lackluster pacing (yes, there's a lot to cover, but the narrative swings between too much happening and sudden lulls, or a weird build-up to a climax that kind of just happens out of nowhere - bang - and it's done).

The lead, Timothee, falls short of charisma (he's genuinely trying, but he seems miscast)

The romance feels totally unconvincing (almost as bad as that romance in Attack of the Clones).

The villain, despite a cool appearance, lacks depth. Yes, he's psychotic and needs to constantly look up beneath a furrowed hairless brow to show us that, but he's not scary or threatening.

Maybe I need to watch it again, but I was disappointed for sure, and I don't really know if it's just me missing something or if everyone's expectations of good films are being watered down, and we're forgetting that good movies get under your skin, whereas this one barely made a scratch."""

print(predict_sentiment(text, model, tokenizer, vocab, device))
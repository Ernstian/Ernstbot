
Pfosten = []
with open("Dataset.txt", "r+", encoding = "utf8") as file:
    line = file.readline()
    while line:
        Pfosten.append(line.strip())
        line = file.readline()

import random
random.seed("Ernstchan regelt!")

random.shuffle(Pfosten)

TrainTestSplit = 90

LengthAll = len(Pfosten)

TraindataNum = int(LengthAll/100*TrainTestSplit)

Trainingsdata = Pfosten[0:TraindataNum]
Testdata = Pfosten[TraindataNum:]

with open("Trainingdata.txt", "w", encoding = "utf-8") as file:
    for Pfosten in Trainingsdata:
        file.write(Pfosten)
        file.write("\n")
with open("Testdata.txt", "w", encoding = "utf-8") as file:
    for Pfosten in Testdata:
        file.write(Pfosten)
        file.write("\n")



train_path = "Trainingdata.txt"
test_path = "Testdata.txt"

from transformers import AutoTokenizer

PreTrainedModel = "Original_Model"
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(PreTrainedModel)
print("Loaded Tokenizer")

from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator
print("Loading Data")
train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
print("Loaded Data")

from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

print("Loading model")
model = AutoModelWithLMHead.from_pretrained(PreTrainedModel)
print("Model loaded")

training_args = TrainingArguments(
    output_dir="./Ernstbot", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)



trainer.train()
trainer.save_model()


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pydoda import Category

## Create an instance of Category
#my_category = Category('semantic', 'animals')
#darija_translation = my_category.get_darija_translation('cat')
#print('Darija for Cat: ' + darija_translation)
## Output: klb
#
## Get the English translation of a word
#english_translation = my_category.get_english_translation('mch')
#print('English for mch: ' + english_translation)


# CSV file with 'darija' and 'eng' columns
class TranslationDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename, encoding='ISO-8859-1')
        self.data.fillna('', inplace=True)  # Replace NaNs with empty strings
        self.darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        darija = self.data.iloc[idx]['darija']
        english = self.data.iloc[idx]['eng']
        darija_ids = self.darija_tokenizer.encode(darija, add_special_tokens=True)
        english_ids = self.english_tokenizer.encode(english, add_special_tokens=True)
        return torch.tensor(darija_ids), torch.tensor(english_ids)

def collate_fn(batch):
    darija_batch, english_batch = zip(*batch)
    darija_batch = pad_sequence(darija_batch, padding_value=0, batch_first=True)
    english_batch = pad_sequence(english_batch, padding_value=0, batch_first=True)
    return darija_batch, english_batch

dataset = TranslationDataset('../sentences.csv')
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)


# Add special tokens to the tokenizer and extend the embedding layer in the model
special_tokens_dict = {'additional_special_tokens': ['<sos>', '<eos>']}
num_added_toks = dataset.english_tokenizer.add_special_tokens(special_tokens_dict)

num_special_tokens = 2  # for <sos> and <eos>
#new_vocab_size = len(dataset.english_tokenizer.vocab) + num_special_tokens

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim + num_special_tokens, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_dim + num_special_tokens, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim + num_special_tokens)
        
    def forward(self, input, hidden, cell):
        
        # Check if the input needs unsqueezing
        # The input should ideally be [batch_size], and we want [batch_size, 1] for LSTM
        if input.dim() == 1:
            input = input.unsqueeze(1)

        embedded = self.embedding(input)

        # Ensure embedded tensor is correctly shaped for the LSTM
        if embedded.dim() == 3 and embedded.shape[1] != 1:
            raise ValueError(f"Unexpected sequence length dimension: {embedded.shape[1]}")

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Squeeze the sequence length dimension out for the linear layer
        # output shape from LSTM with batch_first=True: [batch_size, seq_len, hid_dim]
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell


# Parameters initialization
INPUT_DIM = len(dataset.darija_tokenizer.vocab)
OUTPUT_DIM = len(dataset.english_tokenizer.vocab)
ENC_EMB_DIM = DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)  
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)

sos_token_id = dataset.english_tokenizer.convert_tokens_to_ids('<sos>')
eos_token_id = dataset.english_tokenizer.convert_tokens_to_ids('<eos>')


#def evaluate(model, loader):
#    model.eval()  # Set the model to evaluation mode
#    total_loss = 0
#    with torch.no_grad():  # No need to track gradients during evaluation
#        for darija_tensor, english_tensor in loader:
#            encoder_hidden, encoder_cell = encoder(darija_tensor)
#            decoder_input = torch.full((darija_tensor.shape[0], 1), dataset.english_tokenizer.vocab['<sos>'], dtype=torch.long) # Use start-of-sequence token
#            decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
#            sequence_length = english_tensor.shape[1]
#            loss = 0
#            for t in range(sequence_length):
#                decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
#                decoder_input = english_tensor[:, t].unsqueeze(1)  # Use current target token as next input
#                loss += criterion(decoder_output, english_tensor[:, t])
#            loss /= sequence_length
#            total_loss += loss.item()
#
#    average_loss = total_loss / len(loader)
#    model.train()  # Set the model back to training mode
#    return average_loss
#
#validation_loss = evaluate(encoder, decoder, validation_loader)
#print(f'Validation Loss: {validation_loss}')

def translate(model, sentence, darija_tokenizer, english_tokenizer):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients for inference
        # Encode the sentence using the darija tokenizer
        darija_ids = torch.tensor([darija_tokenizer.encode(sentence, add_special_tokens=True)])
        # Pass the encoded words to the encoder
        hidden, cell = model.encoder(darija_ids)
        
        outputs = []
        
        # Begin with the <sos> token
        input_token = torch.tensor([sos_token_id], dtype=torch.long)
        for _ in range(100):  # maximum sequence length to generate
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            predicted_token_id = output.argmax(1).item()
            
            # Stop if the <eos> token is generated
            if predicted_token_id == eos_token_id:
                break
            
            # Decode the token ID to a word
            predicted_word = english_tokenizer.decode([predicted_token_id], skip_special_tokens=True)
            # Add the predicted word to the outputs, unless it's a padding token
            if predicted_word != english_tokenizer.pad_token:
                outputs.append(predicted_word)
            
            # Prepare the next input token (predicted token becomes the next input)
            input_token = torch.tensor([predicted_token_id], dtype=torch.long)

        return ' '.join(outputs)

class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TranslationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

def load_model(path='../translation_model_after_training.pth'):
    translation_model = TranslationModel(encoder, decoder)
    translation_model.load_state_dict(torch.load(path))
    return translation_model

translation_model = TranslationModel(encoder, decoder)
# Load the model state dictionary
translation_model.load_state_dict(torch.load('../translation_model_after_training.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translation_model.to(device)
# Criterion for calculating loss
criterion = nn.CrossEntropyLoss()
# Define your optimizer here if not already defined
optimizer = torch.optim.Adam(translation_model.parameters(), lr=0.001)

# Number of epochs to train
num_epochs = 10


#for epoch in range(num_epochs):
#    translation_model.train()  
#    for darija_tensor, english_tensor in loader:
#        darija_tensor, english_tensor = darija_tensor.to(device), english_tensor.to(device)
#        optimizer.zero_grad()  # Clear existing gradients
#        
#        # Encode input sequence
#        encoder_hidden, encoder_cell = translation_model.encoder(darija_tensor)
#
#        # Prepare decoder input (start with <sos> tokens for each sequence in the batch)
#        decoder_input = torch.full((darija_tensor.shape[0], 1), sos_token_id, dtype=torch.long).to(device)
#         # Initialize decoder hidden and cell states with encoder's final states
#        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
#
#        loss = 0
#        sequence_length = english_tensor.shape[1]  # Assuming english_tensor is already padded to the max length
#
#        # Iterate through each token in the target sequence
#        for t in range(sequence_length):
#            decoder_output, decoder_hidden, decoder_cell = translation_model.decoder(decoder_input, decoder_hidden, decoder_cell)
#            decoder_input = english_tensor[:, t].unsqueeze(1)
#            print("Decoder output shape:", decoder_output.shape)  # should show something like [25, vocab_size]
#            #print("Target shape:", english_tensor[:, t].shape)  # should show [25]  # Use current target token as next input to the decoder
#            loss += criterion(decoder_output, english_tensor[:, t])  # Calculate and accumulate loss
#
#        loss /= sequence_length  # Average the loss over the sequence length
#        loss.backward()  # Backpropagate the error
#        optimizer.step()  # Update weights
#
#        
#
#    print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # Print loss for each epoch
#
#
#torch.save(translation_model.state_dict(), 'translation_model_after_training.pth')
input_sentence = "bayna homa tay7awlo ib9aw mbrrdin."

##translation_model.encoder.embedding.resize_token_embeddings(len(dataset.english_tokenizer))
#translation_model.decoder.embedding.resize_token_embeddings(len(dataset.english_tokenizer))
    
#translated_sentence = translate(translation_model, input_sentence, dataset.darija_tokenizer, dataset.english_tokenizer)
#print("Translated Sentence:", translated_sentence)


# Candidate and reference sentences
#candidate_sentence = translated_sentence
#reference_sentence = "It ' s apparent they ' re trying to stick to the rules"
#
## Tokenize sentences
#candidate_tokens = candidate_sentence.split()
#reference_tokens = reference_sentence.split()
#
## Compute BLEU score with smoothing function
#smoothie = SmoothingFunction().method4
#bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
#
#print("BLEU Score:", bleu_score)

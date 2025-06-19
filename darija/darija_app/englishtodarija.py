import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers import AutoTokenizer
DarijaBert_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")

# Assuming you have a CSV file with 'darija' and 'eng' columns
class TranslationDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename, encoding='ISO-8859-1')
        self.data.fillna('', inplace=True)  # Replace NaNs with empty strings
        self.darija_tokenizer = DarijaBert_tokenizer  # Using a multilingual model
        self.english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        english = self.data.iloc[idx]['eng']
        darija = self.data.iloc[idx]['darija']
        english_ids = self.english_tokenizer.encode(english, add_special_tokens=True)
        darija_ids = self.darija_tokenizer.encode(darija, add_special_tokens=True)
        return torch.tensor(english_ids), torch.tensor(darija_ids)

def collate_fn(batch):
    darija_batch, english_batch = zip(*batch)
    darija_batch = pad_sequence(darija_batch, padding_value=0, batch_first=True)
    english_batch = pad_sequence(english_batch, padding_value=0, batch_first=True)
    return darija_batch, english_batch

dataset = TranslationDataset('../sentences.csv')
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)


# Add special tokens to the tokenizer and extend the embedding layer in the model
special_tokens_dict = {'additional_special_tokens': ['<sos>', '<eos>']}
num_added_toks = dataset.darija_tokenizer.add_special_tokens(special_tokens_dict)

num_special_tokens = 2  # for <sos> and <eos>

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
        #print("Input shape before embedding:", input.shape)
        
        # Check if the input needs unsqueezing
        # The input should ideally be [batch_size], and we want [batch_size, 1] for LSTM
        if input.dim() == 1:
            input = input.unsqueeze(1)  # Add sequence length dimension

        embedded = self.embedding(input)
        #print("Embedded shape:", embedded.shape)

        # Ensure embedded tensor is correctly shaped for the LSTM
        # It should be [batch_size, seq_len, emb_dim], where seq_len is expected to be 1
        if embedded.dim() == 3 and embedded.shape[1] != 1:
            raise ValueError(f"Unexpected sequence length dimension: {embedded.shape[1]}")

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Squeeze the sequence length dimension out for the linear layer
        # output shape from LSTM with batch_first=True: [batch_size, seq_len, hid_dim]
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell


# Parameters initialization
INPUT_DIM = len(dataset.english_tokenizer.vocab)
OUTPUT_DIM = len(dataset.darija_tokenizer.vocab)
ENC_EMB_DIM = DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)

sos_token_id = dataset.darija_tokenizer.convert_tokens_to_ids('<sos>')
eos_token_id = dataset.darija_tokenizer.convert_tokens_to_ids('<eos>')


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

def translate(model, sentence, english_tokenizer, darija_tokenizer):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients for inference
        # Encode the sentence using the English tokenizer
        english_ids = torch.tensor([english_tokenizer.encode(sentence, add_special_tokens=True)])
        # Pass the encoded words to the encoder
        hidden, cell = model.encoder(english_ids)
        
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
            predicted_word = darija_tokenizer.decode([predicted_token_id], skip_special_tokens=True)
            # Add the predicted word to the outputs, unless it's a padding token
            if predicted_word != darija_tokenizer.pad_token:
                outputs.append(predicted_word)
            
            # Prepare the next input token (predicted token becomes the next input)
            input_token = torch.tensor([predicted_token_id], dtype=torch.long)

        # Detokenize the outputs
        translated_sentence = detokenize(outputs)
        return translated_sentence

class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TranslationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

def load_model(path='../english-darija_translation_model_after_training.pth'):
    translation_model = TranslationModel(encoder, decoder)
    translation_model.load_state_dict(torch.load(path))
    return translation_model

english_translation_model = TranslationModel(encoder, decoder)
# Load the model state dictionary
english_translation_model.load_state_dict(torch.load('../english-darija_translation_model_after_training.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
english_translation_model.to(device)
# Criterion for calculating loss
criterion = nn.CrossEntropyLoss()
# Define your optimizer here if not already defined
optimizer = torch.optim.Adam(english_translation_model.parameters(), lr=0.001)

def detokenize(tokens):
    # Initialize an empty list to store the reconstructed words
    reconstructed_words = []
    
    # Iterate through the tokens
    for token in tokens:
        # If the token starts with "##" and is not the first token in the sentence
        if token.startswith("##") and len(reconstructed_words) > 0:
            # Remove the "##" prefix and append the rest of the token to the previous word
            reconstructed_words[-1] += token[2:]
        else:
            # If the token does not start with "##", it represents the start of a new word
            # Append the token as a new word
            reconstructed_words.append(token)
    
    # Join the reconstructed words into a single sentence
    reconstructed_sentence = " ".join(reconstructed_words)
    return reconstructed_sentence

# Number of epochs to train
num_epochs = 9


#for epoch in range(num_epochs):
#    english_translation_model.train()
#    for english_tensor, darija_tensor in loader:  # Switched the order to match new inputs and outputs
#        english_tensor, darija_tensor = english_tensor.to(device), darija_tensor.to(device)
#
#
#        optimizer.zero_grad()  # Clear existing gradients
#
#        # Encode input sequence, now English
#        encoder_hidden, encoder_cell = english_translation_model.encoder(english_tensor)
#
#        # Prepare decoder input (start with <sos> tokens for each sequence in the batch)
#        decoder_input = torch.full((english_tensor.shape[0], 1), sos_token_id, dtype=torch.long).to(device)
#         # Initialize decoder hidden and cell states with encoder's final states
#        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
#
#        loss = 0
#        sequence_length = darija_tensor.shape[1]  # Assuming darija_tensor is already padded to the max length
#
#        # Iterate through each token in the target sequence, now Darija
#        for t in range(sequence_length):
#            decoder_output, decoder_hidden, decoder_cell = english_translation_model.decoder(decoder_input, decoder_hidden, decoder_cell)
#            decoder_input = darija_tensor[:, t].unsqueeze(1)  # Use current target token as next input
#            print(f'Epoch {epoch+1}')
#            #print("Decoder output shape:", decoder_output.shape)  # should show something like [batch_size, vocab_size]
#            #print("Target shape:", darija_tensor[:, t].shape)  # should show [batch_size]  # Use current target token as next input to the decoder
#            loss += criterion(decoder_output, darija_tensor[:, t])  # Calculate and accumulate loss
#
#        loss /= sequence_length  # Average the loss over the sequence length
#        loss.backward()  # Backpropagate the error
#        optimizer.step()  # Update weights
#
#    print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # Print loss for each epoch
#
#torch.save(english_translation_model.state_dict(), 'english-darija_translation_model_after_training.pth')

input_sentence = "I'll have a coffee"

##translation_model.encoder.embedding.resize_token_embeddings(len(dataset.english_tokenizer))
#translation_model.decoder.embedding.resize_token_embeddings(len(dataset.english_tokenizer))
    
#translated_sentence = translate(english_translation_model, input_sentence, dataset.english_tokenizer, dataset.darija_tokenizer)
#print("Translated Sentence:", translated_sentence)
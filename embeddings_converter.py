import torch
import torch.nn as nn
import torch.optim as optim


class EmbeddingConverter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        activation_between = nn.ReLU()
        decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.net = nn.Sequential(encoder, activation_between, decoder)

    def forward(self, x):
        return self.net(x)


def train_emb_conv(input_embeddings, output_embeddings) -> EmbeddingConverter:

    input_dim = input_embeddings.shape[1]  # Dimension of first embedding
    hidden_dim = 64  # Bottleneck layer
    output_dim = output_embeddings.shape[1]  # Dimension of second embedding

    # Initialize the model
    emb_conv_model = EmbeddingConverter(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()  # Loss based on difference between embeddings
    optimizer = optim.Adam(emb_conv_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1800
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = emb_conv_model(torch.Tensor(input_embeddings))
        loss = criterion(output, torch.Tensor(output_embeddings))
        loss.backward()
        optimizer.step()
    
        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed!")

    return emb_conv_model

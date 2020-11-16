"""SimGNN class and runner."""

import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule

from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histsogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(abstract_features_1, abstract_features_2.permute([0,2,1])).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[:num_nodes[i], :num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args.bins)
            hist = hist/torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)

        return torch.stack(hist_list).view(-1, self.args.bins)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)

        pooled_features_1 = self.attention(abstract_features_1, batch_1)
        pooled_features_2 = self.attention(abstract_features_2, batch_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)

        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1)
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.setup_model()

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.training_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)
        self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False)
        self.nged_matrix = self.training_graphs.norm_ged
        self.real_data_size = self.nged_matrix.size(0)

        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.number_of_labels = self.training_graphs.num_features

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of tuple of source and target batches.
        """
        source_batches = DataLoader(self.training_graphs.shuffle(), batch_size=self.args.batch_size)
        target_batches = DataLoader(self.training_graphs.shuffle(), batch_size=self.args.batch_size)

        return list(zip(source_batches, target_batches))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        return new_data

    def process_batch(self, data):
        """
        Forward pass with a batch of data.
        :param data: Pair of batches that contains source and target graphs.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        data = self.transform(data)
        target = data["target"]
        prediction = self.model(data)
        loss = torch.nn.functional.mse_loss(prediction, target, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()

        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        for i, graph in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([graph]*len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)

            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            prediction = self.model(data)
            scores[i] = torch.nn.functional.mse_loss(prediction, target, reduction='none').detach().numpy()

        norm_ged_mean = np.mean(ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in ground_truth])
        print("BaseLine Error: " + str(round(base_error*1000, 5)) + ".")
        self.model_error = np.mean(scores).item()
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")

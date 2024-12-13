import numpy as np
import pypianoroll as ppr
import os
import sys
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.utils import *
from utils.createDatasetAE import createDatasetAE
from loadModel import loadModel, loadStateDict
from tensorboardX import SummaryWriter


class VAE(nn.Module):
    def __init__(self, embedding_size=100, covariance_param=0.2):
        super(VAE, self).__init__()

        self.embedding_size = embedding_size
        self.covariance_param = covariance_param

        # ENCODER
        self.encode1 = nn.Sequential(
            nn.Conv2d(1,100,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.Conv2d(100,200,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.Conv2d(200,400,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ELU(),
            nn.Conv2d(400,800,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(800),
            nn.ELU()
        )

        self.encode2 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ELU()
        )
        self.encode31 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )
        self.encode32 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )

        # DECODER
        self.decode1 = nn.Sequential(
            nn.Linear(self.embedding_size,400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Linear(400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,2400),
            nn.BatchNorm1d(2400),
            nn.ELU()
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(800,400,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ELU(),
            nn.ConvTranspose2d(400,200,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.ConvTranspose2d(200,100,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.ConvTranspose2d(100,1,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def encoder(self, hEnc):
        #print("ENOCDER")
        hEnc = self.encode1(hEnc)
        hEnc = torch.squeeze(hEnc,3).view(-1,800*3)
        hEnc = self.encode2(hEnc)
        hEnc1 = self.encode31(hEnc)
        hEnc2 = self.encode32(hEnc)
        return hEnc1, hEnc2

    def decoder(self, z):
        #print("DECODER")
        hDec = self.decode1(z)
        hDec = hDec.view(hDec.size(0),800,-1).unsqueeze(2)
        hDec = self.decode2(hDec)
        return hDec

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(self.covariance_param * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):

    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    #beta for disentanglement
    beta = 1e0

    batch = x.size(0)
    cosSim = torch.sum(cos(x.view(batch,-1),recon_x.view(batch,-1)))
    cosSim = batch-cosSim

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= mu.size(0) * mu.size(1)
    return cosSim + (beta * KLD), cosSim, KLD


def train(epoch):
    model.train()
    trainLoss = 0
    cos_sims = 0
    klds = 0
    loss_divider = len(train_loader.dataset)-(len(train_loader.dataset)%batch_size)
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        reconBatch, mu, logvar = model(data)
        loss, cos_sim, kld = loss_function(reconBatch, data, mu, logvar)
        loss.backward()
        trainLoss += loss.item()
        cos_sims += cos_sim.item()
        klds += kld.item()
        optimizer.step()

        weights = []
        for i, f in enumerate(model.parameters()):
            weights.append(f.cpu().data.numpy())

        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average Loss: {:.4f}'.format(
          epoch, trainLoss / loss_divider))

    return trainLoss / loss_divider, cos_sims / loss_divider, \
                            klds / loss_divider, weights, mu


def test(epoch, data_loader, test_set=False, valid_set=False):
    model.eval()
    loss = 0
    cos_sim = 0
    kld = 0
    loss_divider = len(data_loader.dataset)-(len(data_loader.dataset)%batch_size)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.float().to(device)
            reconBatch, mu, logvar = model(data)

            temp_loss, cos_sim_temp, kld_temp = loss_function(reconBatch, data, mu, logvar)
            loss += temp_loss.item()
            cos_sim += cos_sim_temp.item()
            kld += kld_temp.item()

    loss /= loss_divider
    if test_set:
        print('====> Test set loss: {:.4f}'.format(loss))
    elif valid_set:
        print('====> Validation set loss: {:.4f}'.format(loss))

    return loss, cos_sim / loss_divider, kld / loss_divider


if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='VAE settings')
    parser.add_argument("--file_path", default=None,
        help='Path to your MIDI files.')
    parser.add_argument("--validation_path", default=None,
        help='Path to your validation set. You should take this from a different dataset.')
    parser.add_argument("--model_name", default=None,
        help='Path to your validation set. You should take this from a different dataset.')
    parser.add_argument("--checkpoint", default=None, help='Path to last checkpoint. \
        If you trained the checkpointed model on multiple GPUs use the --is_dataParallel flag. \
        Default: None', type=str)
    args = parser.parse_args()

    if not args.file_path:
        print("You have to set the path to your files from terminal with --file_path flag.")
        sys.exit()
    if not args.model_name:
        print("Please set a model name for checkpoints a plots using --model_name\
                    flag.")
        sys.exit()

    ############################################################################
    ############################################################################
    ############################################################################
    # Hyperparameters
    epochs = 50                     # number of epochs you want to train for
    learning_rate = 1e-3            # starting learning rate
    learning_rate_decay = 0.1       # learning rate_decay per epoch
    lr_decay_step = 5               # step size of learning rate decay
    batch_size = 100                # batch size of autoencoder
    log_interval = 500              # Log/show loss per batch
    embedding_size = 100            # size of latent vector
    beat_resolution = 24            # how many ticks per quarter note:
                                        # 24 to process 1 bar at a time 12 for 2 bars
    seq_length = 96                 # how long is one sequence
    covariance_param = 0.5          # value that the logvar is multiplied by
                                        # in reparameterization
                                        # leave as is (==0.5) for unit variance
    model_name = args.model_name
                                    # name for checkpoints / tensorboard
    ############################################################################
    ############################################################################
    ############################################################################

    writer = SummaryWriter(log_dir=('vae_plots/' + model_name))
    writer.add_text("learning_rate", str(learning_rate))
    writer.add_text("learning_rate_decay", str(learning_rate_decay))
    writer.add_text("learning_rate_decay_step", str(lr_decay_step))
    writer.add_text("batch_size", str(batch_size))
    writer.add_text("embedding_size", str(embedding_size))
    writer.add_text("beat_resolution", str(beat_resolution))
    writer.add_text("model_name", model_name)
    writer.add_text("covariance_param", str(covariance_param))

    #create dataset
    if beat_resolution == 12:
        bars = 2
    else:
        bars = 1

    # check if train and test split already exists
    if os.path.isdir(args.file_path + 'train/') and os.path.isdir(args.file_path + 'test/'):
        print("train/ and test/ folder exist!")
        train_dataset = createDatasetAE(args.file_path + 'train/',
                                  beat_res = beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)

        test_dataset = createDatasetAE(args.file_path + 'test/',
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    else:
        print("Only one folder with all files exist, using {}".format(args.file_path))
        dataset = createDatasetAE(args.file_path,
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)
        train_size = int(np.floor(0.95 * len(dataset)))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # validation set
    if args.validation_path:
        print("Path to validation set was set!")
        valid_dataset = createDatasetAE(args.validation_path,
                                  beat_res = beat_resolution,
                                  bars=bars,
                                  seq_length=seq_length,
                                  binarize=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        print("Please use --valiation_path to set path to validation set.")
        sys.exit()

    print("The training set contains {} sequences".format(len(train_dataset)))
    print("The test set contains {} sequences".format(len(test_dataset)))
    if args.validation_path:
        print("The valdiation set contains {} sequences".format(len(valid_dataset)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(embedding_size=embedding_size, covariance_param=covariance_param)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)
    writer.add_text("pytorch model", str(model))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load Checkpoint
    if args.checkpoint:
        print("Trying to load checkpoint...")
        model = loadStateDict(model, args.checkpoint)

    checkpoint_path = 'checkpoints_vae/'
    best_valid_loss = np.inf
    if learning_rate_decay:
        print("Learning rate decay activated!")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=learning_rate_decay)
    for epoch in range(1, epochs + 1):
        if learning_rate_decay:
            scheduler.step()
        #training with plots
        train_loss, cos_sim_train, kld_train, weights, embedding = train(epoch)
        writer.add_scalar('loss/train_loss_epoch', train_loss, epoch)
        writer.add_scalar('loss/train_reconstruction_loss_epoch', cos_sim_train, epoch)
        writer.add_scalar('loss/train_kld_epoch', kld_train, epoch)
        for i, weight in enumerate(weights):
            writer.add_histogram(('weights/weight{}'.format(i)), weight, global_step=epoch)
        writer.add_histogram('embedding', embedding[0], bins='auto', global_step=epoch)

        #test
        test_loss, cos_sim_test, kld_test = test(epoch, test_loader, test_set=True)
        writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
        writer.add_scalar('loss/test_reconstruction_loss_epoch', cos_sim_test, epoch)
        writer.add_scalar('loss/test_kld_epoch', kld_test, epoch)

        #validate
        valid_loss, cos_sim_valid, kld_valid = test(epoch, valid_loader, valid_set=True)
        writer.add_scalar('loss/valid_loss_epoch', valid_loss, epoch)
        writer.add_scalar('loss/valid_reconstruction_loss_epoch', cos_sim_valid, epoch)
        writer.add_scalar('loss/valid_kld_epoch', kld_valid, epoch)

        #save if model better than before based on validation
        if (valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(model.state_dict(),(checkpoint_path + args.model_name + '.pth'))

    writer.close()

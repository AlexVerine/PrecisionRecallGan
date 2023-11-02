import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.data import Dataset

import logging
from functools import partial

from random import shuffle
from PIL import Image
import numpy as np
import os
import os.path

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501



class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * 255 / 128 - 1
            x[:, 1] = x[:, 1] * 255 / 128 - 1
            x[:, 2] = x[:, 2] * 255 / 128 - 1

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp[0].view(x.shape[0],-1)

def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    if version >= (0, 6):
        kwargs['init_weights'] = False

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)




class SourceTargetDataset(Dataset):
    def __init__(self, batch_size, num_samples):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.transform_test = [transforms.ToTensor()]
       
        self.transform_train = transforms.Compose([
            ] + self.transform_test)

        self.source_features = None
        self.target_samples = None 
        self.feat_size = dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        features = InceptionV3([block_idx], normalize_input=True)
        self.net =  DataParallel(features).cuda().eval()

        self.train()

    def precomputeFeatures_ref(self, fname, images):
        self.source_features = torch.Tensor(self.__loadOrPrecomputeFeatures(fname, images))

    def precomputeFeatures(self, images):
        self.target_features = torch.Tensor(self.__precompute(images))
        shuffle(self.source_features)
        shuffle(self.target_features)

        if len(self.source_features) != len(self.target_features):
            raise ValueError(
                'Length of source samples %d must be identical to length of '
                'target samples %d.'
                % (len(self.source_features), len(self.target_features)))
        nsamples = len(self)
        self.coin_flips =  torch.from_numpy(np.random.binomial(1, 0.5, size=[nsamples])).float()


    def __precompute(self, input):
        if isinstance(input, partial):
            feats = self.extract_features_from_sample_function(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extract_features(input)
            else:
                raise TypeError
        else:
            logging.info(type(input))
            raise TypeError
        return feats
    
    def extract_features_from_sample_function(self, sample):
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        features = []
        with torch.no_grad():
          for bi in range(num_batches):
              start = bi * self.batch_size
              end = start + self.batch_size
              batch , _ = sample()
              feature = self.net(batch.cuda())
              features.append(feature.cpu().data.numpy())
        features = np.concatenate(features, axis=0)
        return features[:self.num_samples]
    
    def extract_features(self, images):
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        _, _, height, width = images.shape
        features = []
        with torch.no_grad():
          for bi in range(num_batches):
              start = bi * self.batch_size
              end = start + self.batch_size
              batch = images[start:end]
              feature = self.net(batch.cuda())
              features.append(feature.cpu().data.numpy())
        return np.concatenate(features, axis=0)
    
    def __loadOrPrecomputeFeatures(self, path, samples):
        if os.path.exists(path):
            embeddings = torch.from_numpy(np.load(path))
            return embeddings
        embeddings = self.__precompute(samples)
        print(type(embeddings))
        with open(path, 'wb') as f:
            np.save(f, embeddings)
        return torch.from_numpy(embeddings)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, flip) 
        """

        flip = self.coin_flips[index]

    
        if not self.is_train:
            flip = not flip
        if flip:
            sample = self.target_features[index]
        else:
            sample = self.source_features[index]

        return sample, flip

    def __len__(self):
        return len(self.source_features)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Source Location: {}\n'.format(self.source_folder)
        fmt_str += '    Target Location: {}\n'.format(self.target_folder)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class ClassifierTrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.totalLoss = np.inf
        self.batch_size = self.dataset.batch_size
        self.feat_size = self.dataset.feat_size

    def initClassifier(self):
        nh=128
        self.classifier  = nn.Sequential(
                nn.Linear(self.feat_size, 1, bias=False),
                )

        self.classifier.cuda().train()
    

    def train(self):
        self.totalLoss=0
        for batch_num, (samples, flips) in enumerate(self.train_loader):
            def closure():
                self.optimizer.zero_grad()
                predictions = self.classifier(samples.cuda())
                loss = self.log_loss(predictions.squeeze(), flips.cuda())
                loss.backward()
                self.totalLoss += float(loss)
                return loss

            self.optimizer.step(closure)


    def test(self):
        self.classifier.eval()
        self.dataset.eval()
        error_I = 0
        error_II = 0
        cnt_I = 0
        cnt_II = 0
        for batch_num, (samples, flips) in enumerate(self.train_loader):
            predictions = self.classifier(samples.cuda())
            predictions = (predictions > 0)
            flips = (flips > 0)
            cnt_I += int((flips.cuda() == 0).sum())
            cnt_II += int((flips.cuda() == 1).sum())
            typeI = (predictions.squeeze() == 1) & (flips.cuda() == 0)
            typeII = (predictions.squeeze() == 0) & (flips.cuda() == 1)
            error_I += int(typeI.sum())
            error_II += int(typeII.sum())
        error_I = float(error_I) / float(cnt_I)
        error_II = float(error_II) / float(cnt_II)
        self.classifier.train()
        self.dataset.train()
        error = 0.5*(error_I + error_II)
        self.scheduler.step(error)
        logging.info(f"loss {self.totalLoss}, error {f'({error_I:.2}+{error_II:.2})/2={error:.2}'}, lr {self.optimizer.param_groups[0]['lr']}")
        return self.stopper.step(error)

    def run(self, num_epochs):
        self.initClassifier()
        self.dataset.train()
        self.train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True, num_workers=0)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-1, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, cooldown=3, factor=0.5)
        self.log_loss = torch.nn.BCEWithLogitsLoss()
        for ep in range(num_epochs):
            self.train()

        return self.classifier

class EnsembleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.networks=[]

    def append(self, net):
        self.networks.append(net)

    def forward(self, x):
        preds = []
        for net in self.networks:
            preds.append(net(x))
        return torch.median(torch.stack(preds), dim=0)[0]

def computePRD(source_folder, target_folder, num_angles=1001, num_runs=10, num_epochs=10):
    precisions = []
    recalls = []
    ensemble = EnsembleClassifier()
    dataset = createTrainTestSets(source_folder, target_folder)
    trainer = ClassifierTrainer(dataset, 'inception')
    for k in range(num_runs):
        classifier = trainer.run(num_epochs)
        ensemble.append(classifier)
    precision, recall = estimatePRD(ensemble, trainer.dataset, num_angles)
    return precision, recall

def compute_prd_from_embedding(eval_data, ref_data, num_clusters=20,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):
  """Computes PRD data from sample embeddings.
  The points from both distributions are mixed and then clustered. This leads
  to a pair of histograms of discrete distributions over the cluster centers
  on which the PRD algorithm is executed.
  The number of points in eval_data and ref_data must be equal since
  unbalanced distributions bias the clustering towards the larger dataset. The
  check can be disabled by setting the enforce_balance flag to False (not
  recommended).
  Args:
    eval_data: NumPy array of data points from the distribution to be evaluated.
    ref_data: NumPy array of data points from the reference distribution.
    num_clusters: Number of cluster centers to fit. The default value is 20.
    num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                The default value is 1001.
    num_runs: Number of independent runs over which to average the PRD data.
    enforce_balance: If enabled, throws exception if eval_data and ref_data do
                     not have the same length. The default value is True.
  Returns:
    precision: NumPy array of shape [num_angles] with the precision for the
               different ratios.
    recall: NumPy array of shape [num_angles] with the recall for the different
            ratios.
  Raises:
    ValueError: If len(eval_data) != len(ref_data) and enforce_balance is set to
                True.
  """

  if enforce_balance and len(eval_data) != len(ref_data):
    raise ValueError(
        'The number of points in eval_data %d is not equal to the number of '
        'points in ref_data %d. To disable this exception, set enforce_balance '
        'to False (not recommended).' % (len(eval_data), len(ref_data)))

  eval_data = np.array(eval_data, dtype=np.float64)
  ref_data = np.array(ref_data, dtype=np.float64)
  precisions = []
  recalls = []
  for _ in range(num_runs):
    eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
    precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
    precisions.append(precision)
    recalls.append(recall)
  precision = np.mean(precisions, axis=0)
  recall = np.mean(recalls, axis=0)
  return precision, recall


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
  """Computes F_beta scores for the given precision/recall values.
  The F_beta scores for all precision/recall pairs will be computed and
  returned.
  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
  Args:
    precision: 1D NumPy array of precision values in [0, 1].
    recall: 1D NumPy array of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 1.
    epsilon: Small constant to avoid numerical instability caused by division
             by 0 when precision and recall are close to zero.
  Returns:
    NumPy array of same shape as precision and recall with the F_beta scores for
    each pair of precision/recall.
  Raises:
    ValueError: If any value in precision or recall is outside of [0, 1].
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  return (1 + beta**2) * (precision * recall) / (
      (beta**2 * precision) + recall + epsilon)


def prd_to_max_f_beta_pair(precision, recall, beta=8):
  """Computes max. F_beta and max. F_{1/beta} for precision/recall pairs.
  Computes the maximum F_beta and maximum F_{1/beta} score over all pairs of
  precision/recall values. This is useful to compress a PRD plot into a single
  pair of values which correlate with precision and recall.
  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
  Args:
    precision: 1D NumPy array or list of precision values in [0, 1].
    recall: 1D NumPy array or list of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 8.
  Returns:
    f_beta: Maximum F_beta score.
    f_beta_inv: Maximum F_{1/beta} score.
  Raises:
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  f_beta = np.max(_prd_to_f_beta(precision, recall, beta))
  f_beta_inv = np.max(_prd_to_f_beta(precision, recall, 1/beta))
  return f_beta, f_beta_inv


def estimatePRD(classifier, dataset, num_angles, epsilon=1e-10):
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    dataset.eval()
    classifier.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    toTorch = lambda z: torch.from_numpy(z).unsqueeze(0).cuda()

    with torch.no_grad():
        fValsAndUs = [(float(classifier(Z.cuda())), int(U)) for Z, U in test_loader]
    fVals = [val for val, U in fValsAndUs]
    fVals = [np.min(fVals)-1] + fVals + [np.max(fVals)+1]
    errorRates = []
    for t in fVals:
        fpr=sum([(fOfZ>=t) and U==0 for fOfZ,U in fValsAndUs]) / float(sum([U==0 for fOfZ,U in fValsAndUs]))
        fnr=sum([(fOfZ<t) and U==1 for fOfZ,U in fValsAndUs]) / float(sum([U==1 for fOfZ,U in fValsAndUs]))
        errorRates.append((float(fpr), float(fnr)))
    precision = [] 
    recall = []
    for slope in slopes:
        prec = min([slope*fnr+fpr for fpr,fnr in errorRates])
        precision.append(prec)
        rec =  min([fnr+fpr/slope for fpr,fnr in errorRates])
        recall.append(rec)

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        print(max_val)
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall



class PRCurves():
    def __init__(self, batch_size=50, num_samples=10000):
        self.dataset = SourceTargetDataset(batch_size, num_samples)
        self.num_runs = 10
        self.num_angles = 1001
        self.num_epochs = 10
    def precision_recall_curve(self, sample):
        self.dataset.precomputeFeatures(sample)
        ensemble = EnsembleClassifier()
        trainer = ClassifierTrainer(self.dataset)

        for k in range(self.num_runs):
            classifier = trainer.run(self.num_epochs)
            ensemble.append(classifier)
        precision, recall = estimatePRD(ensemble, self.dataset, self.num_angles)
        prcurve = {"precision": precision, "recall":recall}
        P, R = prd_to_max_f_beta_pair(precision, recall)
        return P, R,  prcurve
     
# This produces a function which takes in an iterator which returns a set number of samples
# and iterates until it accumulates config['num_pr_images'] images.
# The iterator can return samples with a different batch size than used in
# training, using the setting confg['inception_batchsize']
def prepare_pr_curve(config):
    path = 'samples/features/'+config['dataset'].strip('_hdf5')+'_inception_features.npy'
    path_pr_curve = 'logs/'+config['experiment_name']+'/'
    assert os.path.exists(path), (
                'Make sure you have launched calculate_inception_features.py before ! '
                     )
    
    prc = PRCurves(config['batch_size'], num_samples=config['num_pr_images'])
    prc.dataset.precomputeFeatures_ref(path, None)

    def get_pr_metrics(sample,itr, prints=False):
        if prints:
            logging.info('Gathering PR curves.')
        P, R,  prcurve = prc.precision_recall_curve(sample)
        torch.save(prcurve, path_pr_curve+f'prcurve_{itr}.pth')

        return P, R
    
    return get_pr_metrics


import os

import pytest
import torch

from margipose.data.get_dataset import Base_Data_Dir


@pytest.fixture
def skeleton_canonical_univ():
    """Canonical universal skeleton for S1,Seq1,camera0,frame0 of MPI-INF-3DHP."""
    return torch.as_tensor([
        [ -14.1671, -334.8410, 3685.4099],
        [  -1.8908,  -78.7086, 3697.4800],
        [  12.3105,   -6.8914, 3570.3000],
        [  28.6693,   53.3262, 3259.5300],
        [  65.5078,   80.3900, 3018.8301],
        [ -21.9359,    6.5647, 3823.5701],
        [ -48.9321,    9.3914, 4139.3799],
        [ -48.1227,   29.9672, 4383.5200],
        [  26.1703,  404.6510, 3596.6575],
        [ -15.4026,  957.8070, 3670.3301],
        [ -87.2411, 1390.7700, 3718.3999],
        [ -22.8190,  401.2070, 3829.8625],
        [ -45.7490,  956.8290, 3800.5901],
        [-137.3620, 1388.2400, 3780.2000],
        [   1.6757,  402.9290, 3713.2600],
        [ -11.7886,  176.2583, 3705.0913],
        [  11.9904, -164.0930, 3696.2600],
    ])


@pytest.fixture
def skeleton_mpi3d_univ():
    """28-joint universal skeleton for S1,Seq1,camera0,frame0 of MPI-INF-3DHP."""
    return torch.as_tensor([
        [ -26.0276,   98.0811, 3699.6000],
        [ -45.5924,   -6.8788, 3691.5100],
        [ -11.8660,  175.6800, 3705.0600],
        [ -11.7886,  176.2583, 3705.0914],
        [   1.6757,  402.9290, 3713.2600],
        [  -1.8908,  -78.7086, 3697.4800],
        [  11.9904, -164.0930, 3696.2600],
        [ -14.1671, -334.8410, 3685.4100],
        [  10.8534,  -43.9395, 3744.1400],
        [ -21.9359,    6.5647, 3823.5700],
        [ -48.9321,    9.3914, 4139.3800],
        [ -48.1227,   29.9672, 4383.5200],
        [ -57.2134,   51.5208, 4469.8200],
        [  27.8253,  -40.3641, 3662.9000],
        [  12.3105,   -6.8914, 3570.3000],
        [  28.6693,   53.3262, 3259.5300],
        [  65.5078,   80.3900, 3018.8300],
        [  74.3912,   90.6255, 2930.4500],
        [ -22.8190,  401.2070, 3829.8625],
        [ -45.7490,  956.8290, 3800.5900],
        [-137.3620, 1388.2400, 3780.2000],
        [ -43.9510, 1416.6700, 3807.9400],
        [ -17.2509, 1412.7200, 3812.4200],
        [  26.1703,  404.6510, 3596.6575],
        [ -15.4026,  957.8070, 3670.3300],
        [ -87.2411, 1390.7700, 3718.4000],
        [  10.3942, 1414.7500, 3704.4200],
        [  36.6540, 1407.8400, 3701.0500],
    ])


@pytest.fixture
def base_data_dir():
    dir_path = Base_Data_Dir
    if not os.path.isdir(dir_path):
        pytest.skip('base data directory not found')
    return dir_path


@pytest.fixture
def mpi3d_data_dir(base_data_dir):
    dir_path = os.path.join(base_data_dir, 'mpi3d')
    if not os.path.isdir(dir_path):
        pytest.skip('mpi3d data directory not found')
    return dir_path

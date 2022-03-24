from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mattstools.modules import DenseNetwork
from mattstools.network import MyNetBase
from mattstools.torch_utils import move_dev, to_np, get_loss_fn
from mattstools.plotting import plot_multi_hists

from METNetwork.resources.plotting import plot_and_save_hists, plot_and_save_contours


class METNet(MyNetBase):
    """A network for missing transverse momentum reconstruction

    At it's core is a simple and configurable dense network which is
    enveloped in pre- and post-processing layers which perform:
    - masking (shrinking input list)
    - scaling
    - rotations

    These outer layers are disabled during training as our trianing datasets
    are already processed!

    For evaluation and deployement the network requires the following to have been
    stored in its pytorch buffer:
    - inpt_idxes: A list of indices to apply the input mask
    - x_idxes, y_idxes: The indices of the components used for rotations
    - inpt_means, inpt_sdevs: Stats for the input standardisation
    - outp_means, outp_sdevs: Stats for the undoing the standardisation in the output
    """

    def __init__(
        self,
        do_rot: bool,
        n_wpnts: int,
        reg_loss_fn: str = "huber",
        dst_loss_fn: str = "engmmd",
        dst_weight: float = 0,
        base_kwargs: dict = None,
        dense_kwargs: dict = None,
    ):
        """
        args:
            do_rot: If the network should rotate the data on a full pass
            base_kwargs: Kwargs for MyNetBase
            dense_kwargs: Keyword arguments for the dense network
        """
        super().__init__(**base_kwargs)

        ## Save the rotation setting and the loss functions
        self.do_rot = do_rot
        self.n_wpnts = n_wpnts
        self.reg_loss_nm = reg_loss_fn
        self.dst_loss_nm = dst_loss_fn
        self.loss_names += [self.reg_loss_nm, self.dst_loss_nm]
        self.dst_weight = dst_weight
        self.do_dst = self.dst_weight > 0

        ## The actual loss functions
        self.reg_loss_fn = get_loss_fn(self.reg_loss_nm)
        self.dst_loss_fn = get_loss_fn(self.dst_loss_nm)

        ## Initialise the dense network
        self.dense_net = DenseNetwork(
            inpt_dim=self.inpt_dim, outp_dim=2, **dense_kwargs
        )

        ## Move the network to the selected device
        self.to(self.device)

    def get_losses(self, sample):
        """Fill and return the loss dictionary of the sample"""

        ## Unpack the sample
        inputs, targets, weights = sample

        ## Pass through network
        outputs = self.dense_net(inputs)

        ## Calculate the weighted regression loss
        reg_loss = (self.reg_loss_fn(outputs, targets).mean(dim=1) * weights).mean()

        ## Calculate the distance matching loss (if required)
        if self.do_dst:
            dst_loss = self.dst_loss_fn(outputs, targets)
        else:
            dst_loss = T.zeros_like(reg_loss)

        ## Fill the loss dictionary and return
        loss_dict = self.loss_dict_reset()
        loss_dict[self.reg_loss_nm] = reg_loss
        loss_dict[self.dst_loss_nm] = dst_loss
        loss_dict["total"] = reg_loss + self.dst_weight * dst_loss
        return loss_dict

    def full_pass(self, data: T.Tensor) -> T.Tensor:
        """The full pass through the entire METNet layers"""
        data, angles = self.pre_process(data)
        data = self.mlp(data)
        data = self.pst_process(data, angles)
        return data

    def pre_process(self, inpts: T.Tensor) -> T.Tensor:
        """Inputs (all 77 from the tool) are masked, rotated, and scaled"""

        ## Extract the angle of rotation
        angles = inpts[:, -1:]

        ## Apply the mask to the inpts
        inpts = inpts[:, self.inpt_idxes]

        ## Apply the rotations
        if self.do_rot:
            new_x = inpts[:, self.x_idxes] * T.cos(angles) + inpts[
                :, self.y_idxes
            ] * T.sin(angles)
            new_y = -inpts[:, self.x_idxes] * T.sin(angles) + inpts[
                :, self.y_idxes
            ] * T.cos(angles)
            inpts[:, self.x_idxes] = new_x
            inpts[:, self.y_idxes] = new_y

        ## Apply the standardisation
        inpts = (inpts - self.inpt_means) / self.outp_sdevs

        return inpts, angles.squeeze()

    def pst_process(self, output, angles):

        ## Undo the standardisation
        output = output * self.outp_sdevs + self.outp_means

        ## Undo the rotations
        if self.do_rot:
            new_x = output[:, 0] * T.cos(-angles) + output[:, 1] * T.sin(-angles)
            new_y = -output[:, 0] * T.sin(-angles) + output[:, 1] * T.cos(-angles)
            output[:, 0] = new_x
            output[:, 1] = new_y

        return output

    def visualise(self, loader: DataLoader, path: Path, flag: str):
        """Method iterates through a subset of the dataloader and creates performance
        metrics as csvs and images

        The following files are added to the visualise folder

        perf_X.csv: Performance profiles binned in True ET for the following metrics
            - Resolution (x, y)
            - Deviation from linearity
            - Angular Resolution
        The above profiles are also saved as images

        XXX_Dist_X.png: Histograms and contours containing
            - 1D distributions of the reconstructed and true et (processed)
            - 2D distributions of the reconstructed and true x,y (raw and processed)

        args:
            loader: The input dataloader
            path: The visualisation folder
            flag: A flag for visualisation, during training this is the epoch number

        """
        print(f"Running performance profiles on {loader.dataset.n_samples} samples")

        ## Create the subfolder
        out_path = Path(path, f"perf_{flag}")
        out_path.mkdir(parents=True, exist_ok=True)

        ## The bin setup to use for the profiles
        n_bins = 50
        mag_bins = np.linspace(0, 450, n_bins + 1)
        trg_bins = [
            np.linspace(-4, 4, n_bins + 1) + self.do_rot,
            np.linspace(-4, 4, n_bins + 1),
        ]
        exy_bins = [
            np.linspace(-300, 300, n_bins + 1) + self.do_rot * 100,
            np.linspace(-300, 300, n_bins + 1),
        ]

        ## All the networks outputs and targets for the batch combined into one list
        all_outputs = []
        all_targets = []

        ## The information to be saved in our dataframe,
        ## The truth et (for binning) and the performance metric per bin
        met_names = ["Tru", "Res", "Lin", "Ang"]

        ## Configure pytorch, the network and the loader appropriately
        T.set_grad_enabled(False)
        self.eval()
        loader.dataset.weight_off()

        ## Iterate through the validation set
        for i, batch in tqdm(enumerate(loader), desc="perfm", ncols=80, ascii=True):

            ## Get the network outputs and targets (drop weights)
            inputs, targets = move_dev(batch[:-1], self.device)
            outputs = self.dense_net(inputs)

            all_outputs.append(outputs)
            all_targets.append(targets)

        ## Combine the lists into single tensors
        all_outputs = T.cat(all_outputs)
        all_targets = T.cat(all_targets)

        ## Undo the normalisation on the outputs and the targets and convert to GeV
        net_xy = (all_outputs * self.outp_sdevs + self.outp_means) / 1000
        tru_xy = (all_targets * self.outp_sdevs + self.outp_means) / 1000
        net_et = T.norm(net_xy, dim=1)
        tru_et = T.norm(tru_xy, dim=1)

        ## Calculate the performance metrics (ang is through the dot product)
        res = ((net_xy - tru_xy) ** 2).mean(dim=1)
        lin = (net_et - tru_et) / (tru_et + 1e-8)
        ang = T.acos(T.sum(net_xy * tru_xy, dim=1) / (net_et * tru_et + 1e-8)) ** 2

        ## Combine the performance metrics into a single pandas dataframe
        combined = T.vstack([tru_et, res, lin, ang]).T
        df = pd.DataFrame(to_np(combined), columns=met_names)

        ## Make the profiles in bins of True ET using pandas cut and groupby methods
        df["TruM"] = pd.cut(
            df["Tru"], mag_bins, labels=(mag_bins[1:] + mag_bins[:-1]) / 2
        )
        profs = df.drop("Tru", axis=1).groupby("TruM", as_index=False).mean()
        profs["Res"] = np.sqrt(profs["Res"])  ## Res and Ang are RMSE measurements
        profs["Ang"] = np.sqrt(profs["Ang"])

        ## Save the performance profiles
        profs.to_csv(Path(out_path, "perf.csv"), index=False)

        ## Save the profiles as images
        lims = [[0, 50], [-0.2, 0.5], [0, 2.5]]
        for prof_nm, ylim in zip(met_names[1:], lims):
            fig, ax = plt.subplots()
            ax.plot(profs["TruM"], profs[prof_nm])
            ax.set_xlabel("TruM")
            ax.set_ylabel(prof_nm)
            ax.set_ylim(ylim)
            ax.grid()
            fig.savefig(Path(out_path, f"{prof_nm}.png"))
            plt.close(fig)

        ## Save the Magnitude histograms
        h_tru_et = np.histogram(to_np(tru_et), mag_bins, density=True)[0]
        h_net_et = np.histogram(to_np(net_et), mag_bins, density=True)[0]
        plot_and_save_hists(
            Path(out_path, "MagDist"),
            [h_tru_et, h_net_et],
            ["Truth", "Outputs"],
            ["MET Magnitude [Gev]", "Normalised"],
            mag_bins,
            do_csv=True,
        )

        ## Save the target contour plots
        h_tru_tg = np.histogram2d(*to_np(all_targets).T, trg_bins, density=True)[0]
        h_net_tg = np.histogram2d(*to_np(all_outputs).T, trg_bins, density=True)[0]
        plot_and_save_contours(
            Path(out_path, "TrgDist"),
            [h_tru_tg, h_net_tg],
            ["Truth", "Outputs"],
            ["scaled x", "scaled y"],
            trg_bins,
            do_csv=True,
        )

        ## Save the ex and ey contour plots
        h_tru_xy = np.histogram2d(*to_np(tru_xy).T, exy_bins, density=True)[0]
        h_net_xy = np.histogram2d(*to_np(net_xy).T, exy_bins, density=True)[0]
        plot_and_save_contours(
            Path(out_path, "ExyDist"),
            [h_tru_xy, h_net_xy],
            ["Truth", "Outputs"],
            ["METx [GeV]", "METy [GeV]"],
            exy_bins,
            do_csv=True,
        )

        ## Make sure the dataset's weight function is reenabled
        loader.dataset.weight_on()

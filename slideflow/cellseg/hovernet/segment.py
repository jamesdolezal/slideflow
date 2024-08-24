"""Efficient whole-slide HoVerNet segmentation."""

import numpy as np
import requests
import torch
import slideflow as sf
import multiprocessing as mp

from typing import Optional, Union, List, Dict
from torch.nn import functional as F
from os.path import join, basename, exists
from tqdm import tqdm

from slideflow.util import Progress, TextColumn, TileExtractionSpeedColumn, BarColumn, progress
from slideflow.model.torch import autocast
from slideflow.model.torch_utils import get_device
from slideflow.slide.qc.strided_qc import _StridedQC_V2

from .model import HoVerNetPlus

# ----------------------------------------------------------------

def chunk_generator(array, chunk_size):
    rows, cols = array.shape
    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            # Adjust chunk boundaries for edge cases
            end_i = min(i + chunk_size, rows)
            end_j = min(j + chunk_size, cols)
            if end_i - i < chunk_size:
                i = max(0, end_i - chunk_size)
            if end_j - j < chunk_size:
                j = max(0, end_j - chunk_size)

            chunk = array[i:end_i, j:end_j]
            yield (chunk, i, j)

# ----------------------------------------------------------------

class HoVerNetProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            self.columns = (
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                progress.TaskProgressColumn(),
                progress.MofNCompleteColumn(),
                "●",
                TileExtractionSpeedColumn(),
                "●",
                progress.TimeRemainingColumn()
            )
            yield self.make_tasks_table([task])

# ----------------------------------------------------------------


class HoVerNetSegment(_StridedQC_V2):

    # Configuration for the oral epithelial dysplasia model,
    # available at https://github.com/adamshephard/OMTscoring_inference
    OED_CONFIG = {
        'architecture': {
            'class': 'hovernetplus.HoVerNetPlus',
            'kwargs': {'num_layers': 5, 'num_types': 3}
        },
        'ioconfig': {
            'class': 'semantic_segmentor.IOSegmentorConfig',
            'kwargs': {
                'input_resolutions': [{'resolution': 0.5, 'units': 'mpp'}],
                'margin': 128,
                'output_resolutions': [{'resolution': 0.5, 'units': 'mpp'},
                                    {'resolution': 0.5, 'units': 'mpp'},
                                    {'resolution': 0.5, 'units': 'mpp'},
                                    {'resolution': 0.5, 'units': 'mpp'}],
                'patch_input_shape': [256, 256],
                'patch_output_shape': [164, 164],
                'save_resolution': {'resolution': 0.5, 'units': 'mpp'},
                'stride_shape': [164, 164],
                'tile_shape': [2048, 2048]}
        },
        'url': 'https://tiatoolbox.dcs.warwick.ac.uk/models/seg/hovernetplus-oed.pth'
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        *,
        device: str = 'cuda',
        mixed_precision: bool = True,
        batch_size: int = 32,
        persistent_threads: bool = False,
        process_in_chunks: bool = True,
        **kwargs
    ):
        """Create a HoVerNet segmentor for oral epithelial dysplasia."""
        self.config = config or self.OED_CONFIG
        tile_px = self.config['ioconfig']['kwargs']['patch_input_shape'][0]
        tile_um = int(np.round(tile_px / 2)) # 20X magnification
        self.out_px = self.config['ioconfig']['kwargs']['patch_output_shape'][0]
        overlap = int(np.round((tile_px - self.out_px)/2)) + 1

        super().__init__(tile_px=tile_px, tile_um=tile_um, overlap=overlap, persistent_threads=persistent_threads, **kwargs)

        self.out_classes = 0
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.process_in_chunks = process_in_chunks

        # Build the model
        self.device = get_device(device)
        self.model = self.build_model()
        self.model.to(device)
        self.model.eval()

    # ----------------------------------------------------------------
    # Weights & Model
    @staticmethod
    def download_weights(url: str, message: str = "Downloading") -> str:
        """Download a file from the given URL."""
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()

        block_size = 4096
        block_per_mb = block_size / 1000000
        file_size = int(response.headers.get('Content-Length', ''))
        file_size_mb = file_size / 1000000
        running_total_mb = 0
        sf_cache = sf.util.make_cache_dir_path('hovernet')
        file_name = join(sf_cache, basename(url))
        pbar = tqdm(desc=message,
                    total=file_size_mb, unit='MB',
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                            "{n:.2f}/{total:.2f} [{elapsed}<{remaining}] "
                            "{rate_fmt}{postfix}")

        with open(file_name, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=block_size):
                output_file.write(chunk)
                if block_per_mb + running_total_mb < file_size_mb:
                    running_total_mb += block_per_mb  # type: ignore
                    pbar.update(block_per_mb)
                else:
                    running_total_mb += file_size_mb - running_total_mb  # type: ignore
                    pbar.update(file_size_mb - running_total_mb)

        return file_name

    def get_weights(self, url):
        """Get HoverNet weights."""
        weights_path = join(sf.util.make_cache_dir_path('hovernet'), basename(url))
        if exists(weights_path):
            return weights_path
        else:
            return self.download_weights(url)

    def build_model(self, device='cuda'):
        """Build the model."""
        model = HoVerNetPlus(**self.config['architecture']['kwargs'])
        weights = self.get_weights(self.config['url'])
        model.load_state_dict(torch.load(weights, map_location=device), strict=True)
        model.to(device)
        return model

    # ----------------------------------------------------------------
    # Mask functions

    def _calc_mask(self, item):
        """Calculate a QC mask from a given tile."""
        grid_i = item['grid'][0]
        grid_j = item['grid'][1]
        image = item['image']
        mask = self.apply(image)
        return mask, (grid_i, grid_j)

    def _calc_mask_batch(self, batch):
        """Calculate a QC mask from a given tile."""
        batch_images = [item['image'] for item in batch]
        mask = self.apply(batch_images)
        return mask, [(item['grid'][0], item['grid'][1]) for item in batch]

    def build_masks(self, wsi: "sf.WSI"):
        """Return empty arrays for storing QC mask and the average (taper) mask."""
        dim = (wsi.dimensions[1], wsi.dimensions[0])
        px_ratio = wsi.tile_px / wsi.full_extract_px
        target_dim = tuple((np.array(dim) * px_ratio).astype(int))
        nuclear_mask = np.zeros(target_dim, np.int64)
        layer_mask = np.zeros(target_dim, np.int64)
        return nuclear_mask, layer_mask

    def get_pred_bounds(self, wsi: "sf.WSI", i: int, j: int):
        """Return the bounds of the model output for a tile."""
        fy, fx = wsi.grid_to_coord(i, j, anchor="topleft")
        px_ratio = wsi.tile_px / wsi.full_extract_px
        x0 = int(np.round(fx * px_ratio)) + self.overlap
        y0 = int(np.round(fy * px_ratio)) + self.overlap
        x1 = x0 + self.out_px
        y1 = y0 + self.out_px
        return x0, x1, y0, y1

    # ----------------------------------------------------------------
    # HoVerNet functions

    def _preprocess(self, image):
        """HoverNet apparently doesn't do any preprocessing apart from float32 conversion"""
        image = image.to(torch.float32)
        image = sf.io.torch.as_cwh(image)
        return image

    def _postprocess(self, out_dict):
        """Convert model output to nuclear predictions, layer predictions, and instance information."""

        all_out = []
        for i in range(len(out_dict['np'])):
            np_map = out_dict['np'][i]
            hv_map = out_dict['hv'][i]
            tp_map = out_dict['tp'][i]
            ls_map = out_dict['ls'][i]

            # Generate nuclear predictions
            pred_inst = self.model._proc_np_hv(np_map, hv_map, scale_factor=0.5) # fx=0.5 as nuclear processing is at 0.5 mpp instead of 0.25 mpp

            # Generate layer predictions
            # This is delayed until all masks are generated
            pred_layer = ls_map[:, :, 0]

            # Generate nuclear instance information (nuclei type, probability, centroid, and contour for each nucleus)
            pred_type = np.around(tp_map).astype("uint8")
            nuc_inst_info_dict = self.model.get_instance_info(pred_inst, pred_type)

            # Generate layer instance information (contours for each layer)
            # This is delayed until after all masks are generated
            layer_info_dict = {}

            # Append to running
            all_out.append([pred_inst, nuc_inst_info_dict, pred_layer, layer_info_dict])

        return all_out

    def infer_batch(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images."""
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")
        if not image.ndim == 4:
            raise ValueError("Input image must be a 4D tensor of shape (B, C, H, W).")
        if not image.shape[1] == 3:
            raise ValueError("Input image must have 3 channels.")
        if not image.dtype == torch.float32:
            raise ValueError("Input image must be of type float32.")

        image = image.to(self.device)

        with torch.inference_mode():
            # Run inference on a batch of images.
            out_dict = self.model.forward(image)

            # Convert from CWH to WHC
            out_dict = {k: v.permute(0, 2, 3, 1).contiguous() for k, v in out_dict.items()}

            # Apply softmax to nuclear pixels
            out_dict["np"] = F.softmax(out_dict["np"], dim=-1)[..., 1:]

            # Apply softmax to nuclear types
            type_map = F.softmax(out_dict["tp"], dim=-1)
            out_dict["tp"] = torch.argmax(type_map, dim=-1, keepdim=True).to(torch.float32)

            # Don't do anything with the hovernet head ('hv')

            # Apply softmax to the tissue label (epithelium?)
            layer_map = F.softmax(out_dict["ls"], dim=-1)
            out_dict["ls"] = torch.argmax(layer_map, dim=-1, keepdim=True).to(torch.float32)

            # Convert to numpy
            out_dict = {k: v.cpu().to(torch.float32).numpy() for k, v in out_dict.items()}

            # Postprocess and return
            return self._postprocess(out_dict)

    def apply(self, images: Union[np.ndarray, List]) -> np.ndarray:
        if isinstance(images, np.ndarray):
            batch = torch.from_numpy(images)
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
        elif isinstance(images, list):
            batch = torch.stack([torch.from_numpy(img) for img in images])
        else:
            raise TypeError("Input images must be a list or numpy array")
        with torch.inference_mode(), autocast(self.device.type, mixed_precision=self.mixed_precision):
            batch = batch.to(self.device)
            batch = self._preprocess(batch)
            out = self.infer_batch(batch)
            return out

    @property
    def tile_pool(self):
        """Return the tile worker thread pool used for slide reading."""
        return None

    def _post_wsi_load(self, new_wsi, wsi):
        # Transfer QC
        qc_mask = wsi.get_qc_mask(roi=False)
        if qc_mask is not None:
            new_wsi.apply_qc_mask(qc_mask)
        # Transfer ROIs
        if wsi.rois is not None:
            new_wsi.roi_filter_method = wsi.roi_filter_method
            new_wsi.rois = wsi.rois
            new_wsi.roi_method = wsi.roi_method
            new_wsi.process_rois()

    def _post_process_layer(self, layer_class, all_layer_info):
        """Apply the final post-processing, specific to the OED model, removing holes and small objects."""

        # Process layer predictions in chunks. May introduce artifacts.
        if self.process_in_chunks:
            chunk_size = self.config['ioconfig']['kwargs']['tile_shape'][0]
            sf.log.info(f"Processing tissue masks (process_in_chunks=True, chunk_size={chunk_size})...")

            def process_tissue_preds(args):
                chunk, i, j = args
                _layer_info = self.model._get_layer_info(chunk)
                _layer_class = self.model._proc_ls(chunk)
                return _layer_info, _layer_class, i, j

            with mp.dummy.Pool(sf.util.num_cpu()) as pool:
                for _layer_info, _layer_class, i, j in tqdm(pool.imap_unordered(process_tissue_preds,
                                                                                chunk_generator(layer_class, chunk_size)),
                                                            "Finishing up..."):
                    # Override the tissue layer predictions with the post-processed ones
                    layer_class[i:i+chunk_size, j:j+chunk_size] = _layer_class
                    all_layer_info.update(_layer_info)
            return layer_class, all_layer_info

        # Process tissue predictions in one go. May be slow and memory-intensive.
        else:
            sf.log.info(f"Processing tissue masks (process_in_chunks=False...")
            layer_class = self.model._proc_ls(layer_class)
            all_layer_info = self.model._get_layer_info(layer_class)
            return layer_class, all_layer_info

    def _build_progress_bar(self, total):
        """Build the hovernet progress bar."""
        pb = HoVerNetProgress()
        pb.add_task(
            "Extracting tiles...",
            progress_type="tile",
            total=None)
        pred_task = pb.add_task(
            f"Generating preds...",
            progress_type="pred",
            total=total)
        pb.start()
        return pb

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:
        """Apply QC filtering to a slide."""

        # Progress bar tracking
        pb = None if not self.verbose else self._build_progress_bar(wsi.estimated_num_tiles)

        # Load the slide with the correct stride
        qc_wsi, _ = self.get_slide_and_mpp(wsi, pb=pb)
        self._post_wsi_load(qc_wsi, wsi)

        # Prepare empty masks for storing nuclei & tissue segmentations
        nuclear_mask, layer_class = self.build_masks(qc_wsi)
        nuclear_class = np.zeros_like(nuclear_mask, np.int64)

        # Pepare the tile generator, for reading from the WSI in batches
        dts = self.build_tile_generator(qc_wsi)
        dts = sf.util.batch_generator(dts, self.batch_size)

        if self.verbose:
            pb.update(0, total=qc_wsi.estimated_num_tiles)
            pb.update(1, total=qc_wsi.estimated_num_tiles)

        # Apply QC filter to each tile
        if self.filter_pool is not None:
            map_fn = self.filter_pool.imap_unordered
        else:
            map_fn = map

        # Prepare arrays for storing instance information
        all_nuclear_info = {}
        all_layer_info = {}
        running_nuc_count = 0
        running_layer_count = 0

        with sf.util.cleanup_progress(pb):
            for batch_masks, batch_coords in map_fn(self._calc_mask_batch, dts):
                for (tile_masks, (i, j)) in zip(batch_masks, batch_coords):
                    pred_inst, nuc_inst_info, pred_layer, layer_info = tile_masks

                    # Build class masks
                    nuc_class = np.zeros_like(pred_inst)
                    for k, v in nuc_inst_info.items():
                        nuc_class[pred_inst == k] = v['type']

                    # Update values so as not to duplicate IDs
                    pred_inst = pred_inst.astype(np.int64)
                    pred_inst_idx = np.where(pred_inst > 0)
                    pred_inst[pred_inst_idx] += running_nuc_count

                    # Update info dicts
                    nuc_inst_info = {k+running_nuc_count: v for k,v in nuc_inst_info.items()}
                    layer_info = {k+running_layer_count: v for k,v in layer_info.items()}
                    all_nuclear_info.update(nuc_inst_info)
                    all_layer_info.update(layer_info)

                    # Update counts
                    if pred_inst_idx[0].size > 0:
                        running_nuc_count = pred_inst.max()
                    running_layer_count = len(all_layer_info)

                    if pb is not None:
                        pb.advance(1)

                    x0, x1, y0, y1 = self.get_pred_bounds(qc_wsi, i, j)
                    x1 = min(x1, nuclear_mask.shape[0])
                    y1 = min(y1, nuclear_mask.shape[1])
                    nuclear_mask[x0:x1, y0:y1] = pred_inst[0: x1-x0, 0: y1-y0]
                    nuclear_class[x0:x1, y0:y1] = nuc_class[0: x1-x0, 0: y1-y0]
                    layer_class[x0:x1, y0:y1] = pred_layer[0: x1-x0, 0: y1-y0]

        # Process layer predictions
        layer_class, all_layer_info = self.model._post_process_layer(layer_class, all_layer_info)

        if not self.persistent_threads:
            self.close_pools()

        return {
            'nucleus_id': nuclear_mask,
            'nucleus_class': nuclear_class,
            'nucleus_info': all_nuclear_info,
            'tissue_class': layer_class,
            'tissue_info': all_layer_info
        }

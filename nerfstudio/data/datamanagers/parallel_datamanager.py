# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel data manager that generates training data in multiple python processes.
"""
from __future__ import annotations

import concurrent.futures
import queue
import time
import random
from threading import Thread
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Callable,
    Optional,
    Tuple,
    Any,
    Type,
    Union,
)

import torch
import torch.multiprocessing as mp
import psutil
from rich.progress import track
from torch.nn import Parameter

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    VanillaDataManagerConfig,
    TDataset,
    variable_res_collate,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
    PixelSamplerConfig,
    PatchPixelSamplerConfig,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.data_utils import getsizeof_dict
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.comms import get_local_size


@dataclass
class ParallelDataManagerConfig(VanillaDataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

    _target: Type = field(default_factory=lambda: ParallelDataManager)
    """Target class to instantiate."""
    num_processes: int = 1
    """Number of processes to use for train data loading. More than 1 doesn't result in that much better performance"""
    queue_size: int = 2
    """Size of shared data queue containing generated ray bundles and batches.
    If queue_size <= 0, the queue size is infinite."""
    max_thread_workers: Optional[int] = None
    """Maximum number of threads to use in thread pool executor. If None, use ThreadPool default."""


class DataProcessor(mp.Process):
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
    """

    def __init__(
        self,
        dataset: TDataset,
        out_queue: mp.Queue,
        pixel_sampler: PixelSampler,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.dataset = dataset
        self.pixel_sampler = pixel_sampler
        self.collate_fn = collate_fn
        self.num_images_to_sample_from = num_images_to_sample_from
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = False

        if self.num_images_to_sample_from == -1:
            CONSOLE.print("num_images_to_sample_from == -1, evaluate it from dataset size.")
            mem = psutil.virtual_memory()
            dataset_elem_size = getsizeof_dict(self.dataset[0])
            dataset_size = dataset_elem_size * len(dataset)
            CONSOLE.print(f" - Dataset size: {len(dataset)}")
            CONSOLE.print(f" - Dataset element size: {dataset_elem_size/1024**2:.3f} MiB")
            CONSOLE.print(f" - Dataset total size: {dataset_size/1024**3:.3f} GiB")
            CONSOLE.print(f" - Available memory: {mem.available/1024**3:.3f} GiB")
            if self.collate_fn == variable_res_collate:
                CONSOLE.print(
                    "[bold yellow]Warning: variable_res_collate detected, the dataset size estimation may be incorrect."
                )
            if dataset_size * 5 * get_local_size() > mem.available:
                self.num_images_to_sample_from = int(mem.available / 10 / dataset_elem_size / get_local_size())
                if self.num_times_to_repeat_images == -1:
                    CONSOLE.print(
                        f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                        f"resampling at once when cache_images() ends."
                    )
                else:
                    CONSOLE.print(
                        f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                        f"resampling every {self.num_times_to_repeat_images} iters."
                    )
            else:
                CONSOLE.print(f"Caching all {len(dataset)} images.")
                self.cache_all_images = True
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling at once when cache_images() ends."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )
        self.cache_images()

        self.ray_generator = RayGenerator(self.dataset.cameras)
        self.img_data = self.next_img_data
        del self.next_img_data

    def run(self):
        """Append out queue in parallel with ray bundles and batches."""
        count = 0
        if not self.cache_all_images:
            self.cache_images_thread = Thread(target=self.cache_images)
            self.cache_images_thread.start()
        while True:
            batch = self.pixel_sampler.sample(self.img_data)
            ray_indices = batch["indices"]
            ray_bundle: RayBundle = self.ray_generator(ray_indices)
            # check that GPUs are available
            if torch.cuda.is_available():
                ray_bundle = ray_bundle.pin_memory()
            while True:
                try:
                    self.out_queue.put_nowait((ray_bundle, batch))
                    break
                except queue.Full:
                    time.sleep(0.0001)
                except Exception:
                    CONSOLE.print_exception()
                    CONSOLE.print("[bold red]Error: Error occured in parallel datamanager queue.")
            if not self.cache_all_images:
                repeat_image = False
                if self.num_times_to_repeat_images != -1:
                    count += 1
                    repeat_image = count == self.num_times_to_repeat_images
                else:
                    repeat_image = not self.cache_images_thread.is_alive()
                if repeat_image:
                    del self.img_data
                    self.cache_images_thread.join(timeout=60)
                    if self.cache_images_thread.is_alive():
                        raise TimeoutError("Cache images thread timeout! Please increase num_times_to_repeat_images.")

                    self.img_data = self.next_img_data
                    del self.next_img_data, self.cache_images_thread
                    count = 0

                    self.cache_images_thread = Thread(target=self.cache_images)
                    self.cache_images_thread.start()

    def cache_images(self):
        """Caches all input images into a NxHxWx3 tensor."""
        indices = (
            random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
            if not self.cache_all_images
            else range(len(self.dataset))
        )
        batch_list = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # type: ignore
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())
        self.next_img_data = self.collate_fn(batch_list)
        del batch_list, results, indices


class ParallelDataManager(DataManager, Generic[TDataset]):
    """Data manager implementation for parallel dataloading.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def __init__(
        self,
        config: ParallelDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.dataset_type: Type[TDataset] = kwargs.get("_dataset_type", getattr(TDataset, "__default__"))
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        cameras = self.train_dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        # check if spawn is already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        super().__init__()

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training."""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation."""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular, num_rays_per_batch=num_rays_per_batch
        )

    def setup_train(self):
        """Sets up parallel python data processes for training."""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.data_queue = mp.Manager().Queue(maxsize=self.config.queue_size)
        self.data_procs = [
            DataProcessor(
                dataset=self.train_dataset,
                out_queue=self.data_queue,  # type: ignore
                pixel_sampler=self.train_pixel_sampler,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
                collate_fn=self.config.collate_fn,
            )
            for _ in range(self.config.num_processes)
        ]
        for proc in self.data_procs:
            proc.start()
        print("Started threads")

        # Prime the executor with the first batch
        self.train_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers)
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)

    def setup_eval(self):
        """Sets up the data loader for evaluation."""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)  # type: ignore
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the parallel training processes."""
        self.train_count += 1

        # Fetch the next batch in an executor to parallelize the queue get() operation
        # with the train step
        bundle, batch = self.train_batch_fut.result()
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Retrieve the next eval image."""
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def __del__(self):
        """Clean up the parallel data processes."""
        if hasattr(self, "data_procs"):
            for proc in self.data_procs:
                proc.terminate()
                proc.join()

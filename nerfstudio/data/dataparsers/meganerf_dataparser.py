# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# Copyright 2023 Jesse Senior. All rights reserved.
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
Data parser for Mega-NeRF dataset.
"""

import concurrent.futures
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type
from zipfile import ZipFile

import cv2
import torch
import numpy as np
from rich.progress import track

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class MegaNeRFDataParserConfig(DataParserConfig):
    """Mega-NeRF dataset config"""

    _target: Type = field(default_factory=lambda: MegaNeRFDataParser)
    """target class to instantiate"""
    data: Path = Path("/home/nikhil/nerfstudio-main/tests/data/lego_test/minimal_parser")
    """Directory or explicit json file path specifying location of data."""
    train_every: int = 1
    """If set to larger than 1, subsamples each n training images"""
    use_mask: bool = True
    """Use mask for dataset. Automatically disable if masks does not exists"""
    altitude_min: float = -50
    """constrains ray sampling to the given altitude"""
    altitude_max: float = 50
    """constrains ray sampling to the given altitude"""
    scale_factor: float = 5.0
    """How much to scale the camera origins by."""


@dataclass
class MegaNeRFDataParser(DataParser):
    """Mega-NeRF DatasetParser"""

    config: MegaNeRFDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        self.split = split
        coordinate_info = torch.load(self.config.data / "coordinates.pt", map_location="cpu")
        origin_drb = coordinate_info["origin_drb"]
        pose_scale_factor = coordinate_info["pose_scale_factor"]  # Scale factor that already scaled by preprocessing.

        if split == "train":
            image_filenames = list((self.config.data / "train" / "rgbs").iterdir())
            image_filenames = [image_filenames[i] for i in range(0, len(image_filenames), self.config.train_every)]
            image_metadatas = [self.config.data / "train" / "metadata" / f"{x.stem}.pt" for x in image_filenames]
        elif split in ["val", "test"]:
            image_filenames = list((self.config.data / "val" / "rgbs").iterdir())
            image_metadatas = [self.config.data / "val" / "metadata" / f"{x.stem}.pt" for x in image_filenames]
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        mask_filenames = self._get_image_masks(image_filenames) if self.config.use_mask else None

        # extract metadata
        intrinsics = torch.empty((len(image_metadatas), 6))
        camera_to_worlds = torch.empty((len(image_metadatas), 3, 4))
        for i, image_metadata in enumerate(image_metadatas):
            assert image_metadata.exists(), f"Missing image metadata: {image_metadata}"
            metadata = torch.load(image_metadata, map_location="cpu")
            intrinsics[i, :4] = metadata["intrinsics"]
            intrinsics[i, 4] = metadata["H"]
            intrinsics[i, 5] = metadata["W"]
            c2w = metadata["c2w"]  # cam:rub, world:x-down
            camera_to_worlds[i, ...] = torch.cat([c2w[2:3, :], c2w[1:2, :], -c2w[0:1, :]])  # cam:rub, world:z-up

        scale_factor = self.config.scale_factor
        camera_to_worlds[:, :3, 3] *= scale_factor

        # calculate scene box
        scene_box_aabb = (
            torch.stack([torch.min(camera_to_worlds[..., 3], 0)[0], torch.max(camera_to_worlds[..., 3], 0)[0]]) * 1.2
        )
        scene_box_aabb[0, 2] = (self.config.altitude_min - origin_drb[0]) / pose_scale_factor * scale_factor
        scene_box_aabb[1, 2] = (self.config.altitude_max - origin_drb[0]) / pose_scale_factor * scale_factor
        scene_box = SceneBox(aabb=scene_box_aabb)

        cameras = Cameras(
            fx=intrinsics[:, 0].clone(),
            fy=intrinsics[:, 1].clone(),
            cx=intrinsics[:, 2].clone(),
            cy=intrinsics[:, 3].clone(),
            height=intrinsics[:, 4].to(torch.int),
            width=intrinsics[:, 5].to(torch.int),
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        applied_scale = 1 / pose_scale_factor * scale_factor
        applied_transform = torch.eye(4, dtype=torch.float32)[:3, :]
        applied_transform[:, 3] = torch.Tensor([-origin_drb[2], -origin_drb[1], origin_drb[0]])

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_transform=applied_transform,
            dataparser_scale=applied_scale,
            metadata={},
        )
        return dataparser_outputs

    def _get_image_masks(self, image_filenames: List[Path]):
        masks_path: Path = self.config.data / "masks"
        if not masks_path.is_dir():
            CONSOLE.print(f"{self.split} mask not found. Disable mask.")
            return None
        CONSOLE.print("Using dataset mask path: {}".format(masks_path))

        parsed_masks_path: Path = self.config.data / "masks_png"
        if parsed_masks_path.is_dir():
            CONSOLE.print("Mask already parsed, try to load existing masks...", end=" ")
            image_masks = list()
            try:
                for image_filename in image_filenames:
                    image_mask = parsed_masks_path / f"{image_filename.stem}.png"
                    assert image_mask.exists(), f"Missing image mask: {image_mask}"
                    image_masks.append(image_mask)
            except AssertionError as e:
                CONSOLE.print(f"[bold yellow] {e}. Attempt to reparse image masks.")
                shutil.rmtree(parsed_masks_path)
            else:
                CONSOLE.print("Done.")
                return image_masks

        def parse_mask(mask_path):
            with ZipFile(mask_path) as zf:
                with zf.open(mask_path.name) as f:
                    keep_mask = torch.load(f, map_location="cpu")
            keep_mask = keep_mask.numpy().astype(np.uint8) * 255
            cv2.imwrite(str(parsed_masks_path / f"{mask_path.stem}.png"), keep_mask)

        parsed_masks_path.mkdir()
        try:
            thread_pool = list()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # type: ignore
                for mask_path in masks_path.iterdir():
                    res = executor.submit(parse_mask, mask_path)
                    thread_pool.append(res)
                for res in track(thread_pool, description="Parsing image masks", transient=False):
                    res.result()
        except Exception as e:
            shutil.rmtree(parsed_masks_path)
            raise e

        image_masks = list()
        for image_filename in image_filenames:
            image_mask = parsed_masks_path / f"{image_filename.stem}.png"
            assert image_mask.exists(), f"Missing image mask: {image_mask}"
            image_masks.append(image_mask)
        return image_masks

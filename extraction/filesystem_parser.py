"""Filesystem parsing for forensic disk images.

ForenSight ships its own lightweight image format ("FSIM") used by the
controlled corpus generator (doc §6-7): a superblock, an allocation bitmap and
data blocks.  The parser also supports arbitrary raw images (no metadata) and,
optionally, Sleuth Kit (pytsk3) for real FAT/NTFS images when the library is
installed (doc §39 - pytsk3 is optional; the custom processor is the default).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FSIM_MAGIC = b"FSIM0001"
SUPERBLOCK_LAYOUT = "<8sIIIII16s"  # magic, block_size, total_blocks, bitmap_start, bitmap_blocks, cluster_size, volume_id


@dataclass
class ImageInfo:
    path: str
    size_bytes: int
    block_size: int
    total_blocks: int
    fs_type: str                     # FSIM | RAW | TSK
    volume_id: str = ""
    allocated_blocks: Optional[int] = None
    extras: Dict[str, object] = field(default_factory=dict)


class DiskImageParser:
    """Reads a disk image, exposing block-level access and allocation status."""

    def __init__(self, image_path: str | Path, block_size: int = 512):
        self.path = Path(image_path)
        if not self.path.exists():
            raise FileNotFoundError(f"disk image not found: {self.path}")
        self.image_size = self.path.stat().st_size
        self.block_size = block_size
        self._fh = open(self.path, "rb")
        self.info: ImageInfo = self._detect()

    # ------------------------------------------------------------------ core
    def _detect(self) -> ImageInfo:
        self._fh.seek(0)
        head = self._fh.read(512)
        if head[:8] == FSIM_MAGIC:
            _, block_size, total_blocks, bitmap_start, bitmap_blocks, cluster_size, volume_id = \
                struct.unpack_from(SUPERBLOCK_LAYOUT, head, 0)
            return ImageInfo(
                path=str(self.path), size_bytes=self.image_size, block_size=block_size,
                total_blocks=total_blocks, fs_type="FSIM",
                volume_id=volume_id.rstrip(b"\x00").decode("ascii", "replace"),
                extras={"bitmap_start": bitmap_start, "bitmap_blocks": bitmap_blocks,
                        "cluster_size": cluster_size},
            )
        blocks = self.image_size // self.block_size
        return ImageInfo(path=str(self.path), size_bytes=self.image_size,
                         block_size=self.block_size, total_blocks=blocks, fs_type="RAW")

    def block_count(self) -> int:
        return self.info.total_blocks

    def read_block(self, index: int) -> bytes:
        self._fh.seek(index * self.info.block_size)
        data = self._fh.read(self.info.block_size)
        if len(data) < self.info.block_size:
            data = data.ljust(self.info.block_size, b"\x00")
        return data

    def read_range(self, start_block: int, count: int) -> bytes:
        self._fh.seek(start_block * self.info.block_size)
        return self._fh.read(count * self.info.block_size)

    def allocation_bitmap(self) -> Optional[bytearray]:
        """FSIM allocation bitmap (1 bit per block, MSB first); None for RAW images."""
        if self.info.fs_type != "FSIM":
            return None
        start = self.info.extras["bitmap_start"]
        n_blocks = self.info.extras["bitmap_blocks"]
        raw = self.read_range(start, n_blocks)
        total = self.info.total_blocks
        bitmap = bytearray((total + 7) // 8)
        for i in range(total):
            byte_i, bit_i = i // 8, 7 - (i % 8)
            bitmap[byte_i] |= ((raw[byte_i] >> bit_i) & 1) << bit_i
        return bitmap

    def block_allocation_status(self, index: int) -> str:
        """allocated | unallocated | system (header region) | raw."""
        if self.info.fs_type == "FSIM":
            if index < self.info.extras["bitmap_start"] + self.info.extras["bitmap_blocks"]:
                return "system"
            bitmap = self.allocation_bitmap()
            assert bitmap is not None
            byte_i, bit_i = index // 8, 7 - (index % 8)
            return "allocated" if (bitmap[byte_i] >> bit_i) & 1 else "unallocated"
        return "raw"

    def summary(self) -> Dict[str, object]:
        info = dict(
            path=str(self.path), size_bytes=self.image_size, block_size=self.info.block_size,
            total_blocks=self.info.total_blocks, fs_type=self.info.fs_type,
            volume_id=self.info.volume_id,
        )
        if self.info.fs_type == "FSIM":
            bitmap = self.allocation_bitmap()
            assert bitmap is not None
            allocated = sum(bin(b).count("1") for b in bitmap)
            info["allocated_blocks"] = allocated
        return info

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "DiskImageParser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Optional Sleuth Kit integration (doc §39).  Used only when pytsk3 is
# installed AND the image contains a real filesystem; the custom processor
# above is always available as the fallback.
# --------------------------------------------------------------------------

def try_pytsk3_listing(image_path: str | Path) -> Optional[Dict[str, object]]:
    """Best-effort file listing via pytsk3. Returns None when unavailable."""
    try:
        import pytsk3  # type: ignore
    except ImportError:
        return None
    try:
        img = pytsk3.Img_Info(str(image_path))
        fs = pytsk3.TSK_FS_INFO(img)

        def walk(dir_path: str = "/", depth: int = 0) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            if depth > 4:
                return out
            for f in fs.open_dir(dir_path):
                if f.info.meta is None or f.info.name is None:
                    continue
                name = f.info.name.name.decode("utf-8", "replace")
                if name in (".", ".."):
                    continue
                full = dir_path.rstrip("/") + "/" + name
                if f.info.meta.is_dir():
                    out.extend(walk(full, depth + 1))
                else:
                    out.append({"path": full, "size": int(f.info.meta.size),
                                "allocated": bool(f.info.meta.flags & pytsk3.TSK_FS_NAME_FLAG_UNALLOC == 0)})
            return out

        files = walk()
        return {"fs_type": str(fs.info.ftype), "block_size": int(fs.info.block_size),
                "block_count": int(fs.info.block_count), "files": files}
    except Exception:
        return None

"""
AssetScanner - Scans directories for assets
Handles multi-threaded scanning with progress reporting
"""

import os
import threading
from typing import List, Callable, Optional
from queue import Queue

from .asset_item import AssetItem
from ..config import SUPPORTED_EXTENSIONS, MAX_CONCURRENT_SCANS
from ..utils import detect_sequence, begin_sequence_cache, end_sequence_cache
from ..log import log


class AssetScanner:
    """
    Scans directories for supported assets
    
    Features:
    - Recursive directory scanning
    - Multi-threaded for performance
    - Progress callbacks
    - Sequence detection and grouping
    - Cancellable operations
    """
    
    def __init__(self):
        self.is_scanning = False
        self.cancel_requested = False
        self._scan_thread = None
        self._progress_callback = None
        self._complete_callback = None
        self._scanned_sequences = set()  # Track processed sequences
    
    # Number of new/changed assets handed to batch_callback at a time.
    # Results are saved as the scan goes, so a crash or cancel keeps the work done.
    BATCH_SIZE = 200

    def scan_directory(self,
                      directory: str,
                      recursive: bool = True,
                      progress_callback: Optional[Callable] = None,
                      complete_callback: Optional[Callable] = None,
                      batch_callback: Optional[Callable] = None,
                      skip_check: Optional[Callable] = None) -> List[AssetItem]:
        """
        Scan a directory for assets

        Args:
            directory: Path to directory to scan
            recursive: If True, scan subdirectories
            progress_callback: Called with (current, total, filepath)
            complete_callback: Called when scan completes with list of assets
            batch_callback: Called with a list of AssetItem every BATCH_SIZE
                files. When given, assets are not kept in memory after the
                batch is handed over, and the returned list is empty.
            skip_check: Called with a filepath; returning True skips that file
                (used by rescans to leave unchanged files alone)

        Returns:
            List of AssetItem objects (empty in batch mode)
        """
        if not os.path.exists(directory):
            log.warning(f"Directory not found: {directory}")
            return []

        self.is_scanning = True
        self.cancel_requested = False
        self._progress_callback = progress_callback
        self._complete_callback = complete_callback
        self._scanned_sequences = set()

        assets = []
        pending = []   # new/changed assets not yet handed to batch_callback

        begin_sequence_cache()
        try:
            # First pass: collect all files
            all_files = []
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)

                    # Check extension
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        all_files.append(filepath)

                if not recursive:
                    break  # Don't descend into subdirectories

            total_files = len(all_files)

            # Second pass: process files
            for i, filepath in enumerate(all_files):
                if self.cancel_requested:
                    break

                # Report progress. A UI error must never stop the scan.
                if self._progress_callback:
                    try:
                        self._progress_callback(i + 1, total_files, filepath)
                    except Exception as e:
                        log.debug(f"Progress callback failed: {e}")

                try:
                    # Check if this file is part of an already-processed sequence
                    seq_info = detect_sequence(filepath)
                    if seq_info:
                        base_name, frame_num, ext, pattern, padding = seq_info
                        sequence_key = (os.path.dirname(filepath), pattern)

                        if sequence_key in self._scanned_sequences:
                            continue  # Skip, already processed

                        # Mark this sequence as processed
                        self._scanned_sequences.add(sequence_key)
                    elif skip_check and skip_check(filepath):
                        continue  # Unchanged since the last scan

                    # Create asset item
                    asset = AssetItem(filepath)
                except Exception as e:
                    log.warning(f"Error processing {filepath}: {e}")
                    continue

                if batch_callback:
                    pending.append(asset)
                    if len(pending) >= self.BATCH_SIZE:
                        self._flush_batch(batch_callback, pending)
                else:
                    assets.append(asset)

            if batch_callback and pending:
                self._flush_batch(batch_callback, pending)

            # Call completion callback
            if self._complete_callback:
                self._complete_callback(assets)

        finally:
            end_sequence_cache()
            self.is_scanning = False
            self._scanned_sequences.clear()

        return assets

    @staticmethod
    def _flush_batch(batch_callback, pending):
        """Hand the pending assets over and empty the list. Save errors are logged, not fatal."""
        batch = list(pending)
        pending.clear()
        try:
            batch_callback(batch)
        except Exception as e:
            log.warning(f"Saving scan batch failed: {e}")

    def scan_directory_async(self,
                            directory: str,
                            recursive: bool = True,
                            progress_callback: Optional[Callable] = None,
                            complete_callback: Optional[Callable] = None,
                            batch_callback: Optional[Callable] = None,
                            skip_check: Optional[Callable] = None):
        """
        Scan directory in a background thread
        
        Args:
            directory: Path to directory to scan
            recursive: If True, scan subdirectories
            progress_callback: Called with (current, total, filepath)
            complete_callback: Called when scan completes with list of assets
        """
        if self.is_scanning:
            log.debug("Scan already in progress")
            return
        
        def scan_worker():
            self.scan_directory(
                directory,
                recursive,
                progress_callback,
                complete_callback,
                batch_callback=batch_callback,
                skip_check=skip_check,
            )
        
        self._scan_thread = threading.Thread(target=scan_worker, daemon=True)
        self._scan_thread.start()
    
    def scan_multiple_directories(self,
                                  directories: List[str],
                                  recursive: bool = True,
                                  progress_callback: Optional[Callable] = None,
                                  complete_callback: Optional[Callable] = None) -> List[AssetItem]:
        """
        Scan multiple directories
        
        Args:
            directories: List of directory paths
            recursive: If True, scan subdirectories
            progress_callback: Called with (current, total, filepath)
            complete_callback: Called when all scans complete
        
        Returns:
            Combined list of AssetItem objects
        """
        all_assets = []
        
        for i, directory in enumerate(directories):
            if self.cancel_requested:
                break
            
            assets = self.scan_directory(directory, recursive)
            all_assets.extend(assets)
        
        if self._complete_callback:
            self._complete_callback(all_assets)
        
        return all_assets
    
    def scan_files(self, filepaths: List[str]) -> List[AssetItem]:
        """
        Scan specific files (not directories)
        
        Args:
            filepaths: List of file paths
        
        Returns:
            List of AssetItem objects
        """
        assets = []
        processed_sequences = set()
        
        for filepath in filepaths:
            if not os.path.exists(filepath):
                continue
            
            # Check if part of already-processed sequence
            seq_info = detect_sequence(filepath)
            if seq_info:
                base_name, frame_num, ext, pattern, padding = seq_info
                sequence_key = (os.path.dirname(filepath), pattern)
                
                if sequence_key in processed_sequences:
                    continue
                processed_sequences.add(sequence_key)
            
            try:
                asset = AssetItem(filepath)
                assets.append(asset)
            except Exception as e:
                log.warning(f"Error processing {filepath}: {e}")
        
        return assets
    
    def cancel_scan(self):
        """Request cancellation of current scan"""
        self.cancel_requested = True
    
    def is_busy(self) -> bool:
        """Check if scanner is currently scanning"""
        return self.is_scanning
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for current scan to complete"""
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout)


def scan_for_missing_assets(assets: List[AssetItem]) -> List[AssetItem]:
    """
    Check a list of assets and return those that are missing
    
    Args:
        assets: List of AssetItem objects
    
    Returns:
        List of missing assets
    """
    missing = []
    
    for asset in assets:
        if not asset.verify_exists():
            missing.append(asset)
    
    return missing

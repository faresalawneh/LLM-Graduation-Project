"""Startup patch for the AIPerf 0.11.0 dataset reader race (#31).

AIPerf 0.11.0 reads each conversation with seek() and then read() on one memory
map shared by a worker's lookup threads. Another thread can move the cursor in
between, so the read returns another conversation's bytes. This replaces only
that function. It copies mmap[offset:offset+size], which uses no shared cursor.

aiperf_trace_replay.py puts this folder on PYTHONPATH for the AIPerf command
only, so Python imports this file when each AIPerf interpreter starts. Forked
children inherit the patched class. The installed package is not edited.

Fail-closed: if anything is not exactly as expected, the process exits. A plain
raise would not do that, because Python's site module catches the exception
raised by sitecustomize and carries on.

This file shadows any other sitecustomize. On Ubuntu the system one only installs
the apport handler, which the AIPerf virtual environment cannot import.
"""
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import time

REQUIRED_AIPERF_VERSION = "0.11.0"
# SHA-256 of inspect.getsource(MemoryMapDatasetClient.get_conversation) in the
# aiperf 0.11.0 package (dataset/memory_map_utils.py).
TARGET_SOURCE_SHA256 = (
    "fd0054ffca2979425abb29a5be201a794cfa5f71803f568ba6d25637ef9392ee")
# Where each process records that it is patched. Required.
STATUS_FILE_VAR = "AIPERF_PATCH_STATUS_FILE"
PATCH_MARKER = "_inferscope_patch"
FAILURE_EXIT_CODE = 78


class HookError(RuntimeError):
    """The patch cannot be applied exactly as intended."""


def file_sha256(path):
    """Return the SHA-256 of a file."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_slice(data_mmap, offset, size):
    """Return exactly size bytes at offset without using the map's cursor.

    Args:
        data_mmap: The conversation data memory map.
        offset: First byte of the conversation.
        size: Length of the conversation in bytes.

    Returns:
        A copy of the bytes, or None if the range is not fully inside the map.
        A slice would otherwise shorten or wrap such a range without an error.
    """
    if offset < 0 or size < 0 or offset + size > len(data_mmap):
        return None
    return data_mmap[offset:offset + size]


def build_get_conversation(module, record_read):
    """Build the replacement for MemoryMapDatasetClient.get_conversation.

    It is the 0.11.0 function with the seek() and read() pair replaced.

    Args:
        module: aiperf.dataset.memory_map_utils.
        record_read: Called after every successful copy.
    """
    def get_conversation(self, conversation_id):
        if conversation_id not in self.index.offsets:
            raise KeyError(f"Conversation '{conversation_id}' not found in dataset")

        offset_info = self.index.offsets[conversation_id]

        try:
            conv_bytes = read_slice(
                self.data_mmap, offset_info.offset, offset_info.size)
            if conv_bytes is None:
                raise module.MemoryMapSerializationError(
                    f"Conversation data out of range: offset={offset_info.offset}, "
                    f"size={offset_info.size}, file={len(self.data_mmap)} bytes")
            record_read()

            module._logger.debug(
                lambda: f"Loading conversation '{conversation_id}': offset={offset_info.offset}, size={offset_info.size} bytes"
            )

            return self._deserialize_conversation(conv_bytes)

        except (OSError, module.MemoryMapSerializationError) as e:
            module._logger.error(
                f"Failed to load conversation '{conversation_id}' from {self.data_file_path}: {e}"
            )
            raise

    setattr(get_conversation, PATCH_MARKER, True)
    return get_conversation


def _status_writer(status_file, identity):
    """Return a function that appends one event line for this process."""
    def write(event):
        line = json.dumps({"event": event, "pid": os.getpid(),
                           "time": time.time(), **identity}) + "\n"
        descriptor = os.open(status_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)
    return write


def install(module, version, status_file, expected_sha256=TARGET_SOURCE_SHA256):
    """Check the target and replace it.

    Args:
        module: aiperf.dataset.memory_map_utils.
        version: Installed aiperf version.
        status_file: File that receives one line per event and process.
        expected_sha256: SHA-256 the target's source must have.

    Raises:
        HookError: If the version, the target or its source differs, or no
            status file is given. Nothing is patched in that case.
    """
    if version != REQUIRED_AIPERF_VERSION:
        raise HookError(
            f"aiperf {version} is installed; this patch is only for "
            f"{REQUIRED_AIPERF_VERSION}")
    client = getattr(module, "MemoryMapDatasetClient", None)
    target = getattr(client, "get_conversation", None)
    if target is None:
        raise HookError("MemoryMapDatasetClient.get_conversation was not found")
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError) as exc:
        raise HookError(f"cannot read the target's source: {exc}") from exc
    found = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if found != expected_sha256:
        raise HookError(
            f"get_conversation source sha256 {found} != expected {expected_sha256}")
    if not status_file:
        raise HookError(f"{STATUS_FILE_VAR} is not set")

    write = _status_writer(status_file, {
        "aiperf_version": version,
        "target_source_sha256": found,
        "hook_sha256": file_sha256(__file__),
    })
    reported = set()

    def record_read():
        pid = os.getpid()
        if pid not in reported:
            reported.add(pid)
            write("patched_read")

    write("installed")
    client.get_conversation = build_get_conversation(module, record_read)
    os.register_at_fork(after_in_child=lambda: write("forked"))


def _install_or_exit():
    """Patch this interpreter, or end it."""
    try:
        from aiperf.dataset import memory_map_utils

        install(memory_map_utils, importlib.metadata.version("aiperf"),
                os.environ.get(STATUS_FILE_VAR))
    except BaseException as exc:  # nothing may start unpatched
        sys.stderr.write(
            f"AIPerf reader patch FAILED, not starting: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        os._exit(FAILURE_EXIT_CODE)


# Only Python's own startup import patches. Tests load this file by another name.
if __name__ == "sitecustomize":
    _install_or_exit()

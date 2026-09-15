"""Static GPU type tables for remote providers.

Each remote provider registers a table of the GPU types it offers, with nominal
memory and an estimated hourly price. ``table_hardware`` turns one entry into a
hardware description with the same shape as the local ``hardware()`` probe, so
starting defaults work from the table's memory figure unchanged. A provider's
own probe can replace the table figure once a container has run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuType:
    """One GPU type that a provider offers.

    Attributes:
        name: The provider's GPU type string, as its API accepts it.
        label: The GPU model name.
        vram_gb: Nominal memory in decimal gigabytes, from the vendor specification.
        usd_per_hour: Estimated on-demand price for one GPU, in US dollars per hour.
    """

    name: str
    label: str
    vram_gb: int
    usd_per_hour: float

    @property
    def total_mib(self) -> int:
        """Return the memory in MiB, converted from nominal decimal gigabytes.

        Drivers report slightly more memory than this for every listed card, so
        planning from this figure errs on the small side.
        """
        return self.vram_gb * 10**9 // 2**20

    @property
    def vram_gib(self) -> float:
        """Return the memory in GiB, rounded down to one decimal place."""
        return self.total_mib * 10 // 1024 / 10


# Prices from https://modal.com/pricing and GPU type strings from
# https://modal.com/docs/guide/gpu, checked 2026-09-15. Modal's docs state memory
# only for L40S, A100 and H200; the other figures are NVIDIA specifications.
# Modal may run an "H100" request on an H200. "B300" is left out because Modal
# requires CUDA 13.1 or later for it. Prices change: check current Modal pricing.
MODAL_GPUS: tuple[GpuType, ...] = (
    GpuType("T4", "NVIDIA T4", 16, 0.590),
    GpuType("L4", "NVIDIA L4", 24, 0.799),
    GpuType("A10", "NVIDIA A10", 24, 1.102),
    GpuType("L40S", "NVIDIA L40S", 48, 1.951),
    GpuType("A100-40GB", "NVIDIA A100 40GB", 40, 2.099),
    GpuType("A100-80GB", "NVIDIA A100 80GB", 80, 2.498),
    GpuType("RTX-PRO-6000", "NVIDIA RTX PRO 6000 Blackwell", 96, 3.031),
    GpuType("H100", "NVIDIA H100", 80, 3.949),
    GpuType("H200", "NVIDIA H200", 141, 4.540),
    GpuType("B200", "NVIDIA B200", 180, 6.250),
)

MODAL_PRICING_CAVEAT = (
    "Costs are estimates from Modal's published per-GPU prices, checked 2026-09-15, "
    "and exclude CPU, memory and storage charges. Check current Modal pricing at "
    "https://modal.com/pricing."
)

GPU_TABLES: dict[str, tuple[GpuType, ...]] = {"modal": MODAL_GPUS}

PRICING_CAVEATS: dict[str, str] = {"modal": MODAL_PRICING_CAVEAT}


def register_gpu_table(
    provider: str, table: tuple[GpuType, ...], caveat: str = ""
) -> None:
    """Register or replace the GPU type table for a provider.

    Args:
        provider: The provider name. It becomes the hardware ``source``.
        table: The GPU types the provider offers.
        caveat: The text that the panel and CLI show next to cost estimates.
            Empty uses a generic caveat.

    Raises:
        ValueError: If the provider name is ``"local"`` or the table is empty.
    """
    if provider == "local" or not table:
        raise ValueError("A provider table needs a non-local name and GPU types.")
    GPU_TABLES[provider] = tuple(table)
    PRICING_CAVEATS[provider] = caveat or (
        f"Costs are estimates from {provider}'s published prices. "
        "Check current pricing with the provider."
    )


def pricing_caveat(provider: str) -> str:
    """Return the text to show next to a provider's cost estimates.

    Args:
        provider: A registered provider name.

    Returns:
        A sentence that says the cost is an estimate and where to check prices.

    Raises:
        ValueError: If no table is registered for the provider.
    """
    gpu_types(provider)
    return PRICING_CAVEATS[provider]


def gpu_types(provider: str) -> tuple[GpuType, ...]:
    """Return the GPU types that a provider offers.

    Args:
        provider: A registered provider name.

    Returns:
        The provider's GPU types in table order.

    Raises:
        ValueError: If no table is registered for the provider.
    """
    try:
        return GPU_TABLES[provider]
    except KeyError:
        raise ValueError(f"Unknown GPU provider: {provider}.") from None


def gpu_type(provider: str, name: str) -> GpuType:
    """Return one GPU type from a provider's table.

    Args:
        provider: A registered provider name.
        name: The provider's GPU type string.

    Returns:
        The matching GPU type.

    Raises:
        ValueError: If the provider or GPU type is unknown.
    """
    for entry in gpu_types(provider):
        if entry.name == name:
            return entry
    raise ValueError(f"Unknown {provider} GPU type: {name}.")


def table_hardware(provider: str, name: str) -> dict:
    """Build a hardware description for one remote GPU from the table.

    Args:
        provider: A registered provider name.
        name: The provider's GPU type string.

    Returns:
        A description in the ``discovery.hardware()`` shape with one GPU. Host
        memory is unknown, and ``source`` is the provider name.

    Raises:
        ValueError: If the provider or GPU type is unknown.
    """
    entry = gpu_type(provider, name)
    return {
        "gpus": [
            {
                "index": "0",
                "uuid": f"{provider}:{entry.name}",
                "name": entry.label,
                "total_mib": entry.total_mib,
                "used_mib": 0,
                "driver": None,
            }
        ],
        "error": None,
        "ram_gib": None,
        "ram": {"total_gib": None, "used_gib": None, "available_gib": None},
        "source": provider,
    }
